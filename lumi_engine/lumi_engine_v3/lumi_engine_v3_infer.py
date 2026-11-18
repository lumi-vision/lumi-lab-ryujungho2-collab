"""LUMI-7 실시간 스트리밍 추론 및 시연 영상 생성 모듈.

보안 및 무결성 검증:
    1. 체크포인트에 저장된 extractor_config(버전, flash_thresh, ema_tau, ref_fps)와
       검증 상태(validated=True, trained_on_dummy=False)를 확인한다. 구버전(v1) 가중치나
       더미 학습 가중치는 기본적으로 거부된다.
    2. 안전 분석과 보정은 항상 원본 프레임(src_fps) 전부에 대해 수행하여, 낮은 target_fps로
       재표본화할 때 짧은 교번 섬광 프레임이 분석 전에 버려지는 문제를 방지한다.
    3. 출력 컨테이너를 .mp4로 고정하고 VideoWriter.isOpened(), 분석/출력 프레임 수,
       출력 파일 크기(> 0 bytes)를 검증한다.
"""
import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from features import CausalAuxExtractor
from model import LUMI7


class LUMIInferencer:
    """단일 비디오 스트리밍 추론기."""

    def __init__(
        self,
        weights_path: Optional[str] = "lumi7_best.pth",
        device: Optional[str] = None,
        allow_untrained: bool = False,
        allow_unvalidated: bool = False,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor = CausalAuxExtractor().to(self.device).eval()
        self.model = LUMI7(aux_channels=9).to(self.device).eval()
        self.weights_loaded = False
        self.validated = False
        self.trained_fps: Optional[float] = None
        self.weights_path = str(weights_path) if weights_path else None

        if weights_path and Path(weights_path).is_file():
            raw_obj = torch.load(weights_path, map_location=self.device, weights_only=True)
            self._load_and_verify_checkpoint(raw_obj, allow_unvalidated=allow_unvalidated)
        elif not allow_untrained:
            raise FileNotFoundError(
                f"학습된 모델 체크포인트를 찾을 수 없습니다: {weights_path}. "
                "먼저 실제 데이터셋으로 train.py를 실행해 체크포인트를 생성하세요."
            )

    def _load_and_verify_checkpoint(self, ckpt: object, allow_unvalidated: bool) -> None:
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                "체크포인트에 모델 메타데이터 및 extractor_config가 없습니다 (구버전 v1 형식). "
                "v2 특징 추출기 설정으로 재학습된 체크포인트를 사용하세요."
            )

        if "extractor_config" not in ckpt:
            raise ValueError(
                "체크포인트에 extractor_config가 누락되었습니다. v2 추출기 설정과 호환되지 않습니다."
            )

        saved_ext_cfg = ckpt["extractor_config"]
        expected_ext_cfg = self.extractor.get_config()
        if saved_ext_cfg != expected_ext_cfg:
            raise ValueError(
                "특징 추출기 설정 불일치로 체크포인트를 로드할 수 없습니다.\n"
                f"  체크포인트 설정: {saved_ext_cfg}\n"
                f"  현재 추출기 설정: {expected_ext_cfg}"
            )

        saved_model_cfg = ckpt.get("model_config")
        if saved_model_cfg is not None and saved_model_cfg != self.model.get_config():
            raise ValueError(
                f"모델 아키텍처 설정 불일치: 체크포인트={saved_model_cfg}, 현재={self.model.get_config()}"
            )

        meta = ckpt.get("metadata", {})
        is_dummy = bool(meta.get("trained_on_dummy", True))
        is_validated = bool(meta.get("validated", False)) and not is_dummy

        if not is_validated and not allow_unvalidated:
            raise RuntimeError(
                "더미 데이터로 학습되었거나 안전성이 검증되지 않은(validated=False) 체크포인트입니다. "
                "실제 데이터셋으로 학습된 체크포인트만 API 및 안전 판정에 사용할 수 있습니다."
            )

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.weights_loaded = True
        self.validated = is_validated
        self.trained_fps = float(meta["trained_fps"]) if "trained_fps" in meta else None

    @staticmethod
    def _open_mp4_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
        """출력 경로를 .mp4로 강제하고 사용 가능한 코덱으로 VideoWriter를 열어 검증한다."""
        path = Path(output_path)
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"출력 파일은 반드시 .mp4 확장자여야 합니다: {output_path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        for codec in ("avc1", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer
            writer.release()

        raise RuntimeError(
            f"VideoWriter 초기화에 실패했습니다 (path={path}, fps={fps}, size={width}x{height})."
        )

    @staticmethod
    def _verify_output_file(output_path: str) -> None:
        path = Path(output_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"출력 영상 파일이 비어 있거나 생성되지 않았습니다: {output_path}")

    @torch.no_grad()
    def process_video(
        self,
        input_video_path: str,
        output_safe_path: str,
        output_mask_path: Optional[str] = None,
        risk_threshold: float = 0.5,
        target_fps: Optional[float] = None,
        allow_untrained: bool = False,
    ) -> dict:
        """모든 원본 프레임을 분석·보정한 뒤, 지정된 출력 FPS로 인코딩해 진단 리포트를 반환한다."""
        if not self.weights_loaded and not allow_untrained:
            raise RuntimeError(
                "학습된 가중치가 로드되지 않은 상태에서는 영상 보정 추론을 수행할 수 없습니다."
            )

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"입력 영상을 열 수 없습니다: {input_video_path}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0.0 or np.isnan(src_fps):
            src_fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"입력 영상의 해상도가 유효하지 않습니다: {width}x{height}")

        output_fps = float(target_fps) if target_fps and target_fps > 0 else src_fps
        out_safe = self._open_mp4_writer(output_safe_path, output_fps, width, height)
        out_mask = (
            self._open_mp4_writer(output_mask_path, output_fps, width, height)
            if output_mask_path
            else None
        )

        aux_state = None
        model_state = None

        analyzed_frames = 0
        written_frames = 0
        psnr_list = []
        risk_events = []
        start_time = time.perf_counter()

        # 원본 전체 프레임을 분석한 뒤 출력 프레임레이트로 매핑하기 위한 시간축 상태
        src_dt = 1.0 / src_fps
        dst_dt = 1.0 / output_fps
        next_emit_time = 0.0
        peak_mask_np: Optional[np.ndarray] = None

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb_tensor = (
                    torch.from_numpy(frame_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                    / 255.0
                )

                # 1. 모든 원본 프레임에 대해 실제 src_fps 기준으로 특징 추출 및 안전 보정 수행
                aux, aux_state = self.extractor.step(rgb_tensor, aux_state, fps=src_fps)
                out = self.model.step(rgb_tensor, aux, model_state)
                model_state = out["state"]

                safe_rgb = (
                    (out["corrected"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                safe_bgr = cv2.cvtColor(safe_rgb, cv2.COLOR_RGB2BGR)

                mask_np = out["mask"].squeeze().cpu().numpy()
                active_ratio = float((mask_np > risk_threshold).mean())

                if active_ratio > 0.05:
                    sec = analyzed_frames / src_fps
                    risk_events.append(
                        {
                            "frame": analyzed_frames,
                            "timestamp": f"{int(sec // 60):02d}:{sec % 60:05.2f}",
                            "affected_area_ratio": round(active_ratio, 4),
                        }
                    )

                mse = float(
                    np.mean((frame_rgb.astype(np.float32) - safe_rgb.astype(np.float32)) ** 2)
                )
                psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 1e-6 else 50.0
                psnr_list.append(psnr)

                if out_mask is not None:
                    peak_mask_np = (
                        mask_np if peak_mask_np is None else np.maximum(peak_mask_np, mask_np)
                    )

                # 2. 보정 완료된 프레임을 output_fps 시간 격자에 맞춰 출력 (위험 프레임 사전 폐기 없음)
                frame_end_time = (analyzed_frames + 1) * src_dt
                while next_emit_time < frame_end_time - 1e-9:
                    out_safe.write(safe_bgr)
                    if out_mask is not None and peak_mask_np is not None:
                        mask_u8 = (peak_mask_np * 255.0).clip(0, 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(frame_bgr, 0.7, heatmap, 0.3, 0)
                        out_mask.write(overlay)
                    written_frames += 1
                    next_emit_time += dst_dt
                    peak_mask_np = None

                analyzed_frames += 1
        finally:
            cap.release()
            out_safe.release()
            if out_mask is not None:
                out_mask.release()

        if analyzed_frames == 0 or written_frames == 0:
            raise RuntimeError(
                f"영상 처리 프레임 수가 유효하지 않습니다 (analyzed={analyzed_frames}, written={written_frames})."
            )

        self._verify_output_file(output_safe_path)
        if output_mask_path:
            self._verify_output_file(output_mask_path)

        elapsed = max(time.perf_counter() - start_time, 1e-6)

        return {
            "analyzed_frames": analyzed_frames,
            "total_frames": written_frames,
            "source_fps": round(src_fps, 2),
            "fps": round(output_fps, 2),
            "processing_fps": round(analyzed_frames / elapsed, 2),
            "avg_psnr_db": round(float(np.mean(psnr_list)) if psnr_list else 0.0, 2),
            "detected_event_count": len(risk_events),
            "sample_events": risk_events[:10],
            "weights_loaded": self.weights_loaded,
            "validated": self.validated,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI-7 Video Inference Demo")
    parser.add_argument("--input", type=str, required=True, help="입력 영상 경로")
    parser.add_argument("--output-safe", type=str, default="outputs/safe.mp4", help="보정 영상 출력 경로 (.mp4)")
    parser.add_argument("--output-mask", type=str, default="outputs/mask.mp4", help="마스크 오버레이 출력 경로 (.mp4)")
    parser.add_argument("--weights", type=str, default="lumi7_best.pth", help="학습된 모델 체크포인트 경로")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="출력 비디오 재표본화 FPS (안전 분석은 항상 원본 전체 프레임 기준으로 수행)",
    )
    args = parser.parse_args()

    inferencer = LUMIInferencer(weights_path=args.weights, allow_untrained=False)
    report = inferencer.process_video(
        input_video_path=args.input,
        output_safe_path=args.output_safe,
        output_mask_path=args.output_mask,
        target_fps=args.target_fps,
    )
    print("[INFERENCE COMPLETE]")
    print(report)


if __name__ == "__main__":
    main()
