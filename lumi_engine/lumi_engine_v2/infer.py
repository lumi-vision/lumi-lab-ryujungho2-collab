"""LUMI-7 실시간 스트리밍 추론 및 시연 영상 생성 모듈.

보안 및 무결성 검증:
    1. allow_untrained=False(기본값)일 때 학습된 가중치 파일이 없으면 즉시 예외를 발생시켜
       미학습 모델의 무작위 출력이 안전 처리 결과로 반환되는 것을 원천 차단한다.
    2. 출력 컨테이너를 .mp4로 강제하고 VideoWriter.isOpened(), 처리된 프레임 수,
       출력 파일 크기(> 0 bytes)를 검증하여 코덱/파일 생성 실패를 조용히 넘기지 않는다.
    3. 영상의 실제 FPS(또는 지정된 target_fps 재표본화 속도)를 특징 추출기에 전달한다.
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
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor = CausalAuxExtractor().to(self.device).eval()
        self.model = LUMI7(aux_channels=9).to(self.device).eval()
        self.weights_loaded = False
        self.weights_path = str(weights_path) if weights_path else None

        if weights_path and Path(weights_path).is_file():
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True
        elif not allow_untrained:
            raise FileNotFoundError(
                f"학습된 모델 가중치를 찾을 수 없습니다: {weights_path}. "
                "먼저 train.py로 가중치를 학습·저장한 뒤 추론을 실행하세요."
            )

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
        """MP4 파일을 프레임 단위로 처리해 보정 영상과 진단 통계를 반환한다."""
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

        effective_fps = float(target_fps) if target_fps and target_fps > 0 else src_fps
        out_safe = self._open_mp4_writer(output_safe_path, effective_fps, width, height)
        out_mask = (
            self._open_mp4_writer(output_mask_path, effective_fps, width, height)
            if output_mask_path
            else None
        )

        aux_state = None
        model_state = None

        processed_frames = 0
        psnr_list = []
        risk_events = []
        start_time = time.perf_counter()

        # target_fps 지정 시 일정 시간 간격으로 리샘플링하기 위한 누적기
        src_dt = 1.0 / src_fps
        dst_dt = 1.0 / effective_fps
        time_acc = 0.0

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if target_fps is not None:
                    time_acc += src_dt
                    if time_acc + 1e-9 < dst_dt:
                        continue
                    repeats = max(1, int(time_acc // dst_dt))
                    time_acc -= repeats * dst_dt
                else:
                    repeats = 1

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb_tensor = (
                    torch.from_numpy(frame_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(self.device)
                    / 255.0
                )

                for _ in range(repeats):
                    # 1. 실제 FPS 기반 인과적 특징 추출 -> 2. 탐지 및 국소 보정
                    aux, aux_state = self.extractor.step(
                        rgb_tensor, aux_state, fps=effective_fps
                    )
                    out = self.model.step(rgb_tensor, aux, model_state)
                    model_state = out["state"]

                    safe_rgb = (
                        (out["corrected"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    safe_bgr = cv2.cvtColor(safe_rgb, cv2.COLOR_RGB2BGR)
                    out_safe.write(safe_bgr)

                    mask_np = out["mask"].squeeze().cpu().numpy()
                    active_ratio = float((mask_np > risk_threshold).mean())

                    if active_ratio > 0.05:
                        sec = processed_frames / effective_fps
                        risk_events.append(
                            {
                                "frame": processed_frames,
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
                        mask_u8 = (mask_np * 255.0).clip(0, 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(frame_bgr, 0.7, heatmap, 0.3, 0)
                        out_mask.write(overlay)

                    processed_frames += 1
        finally:
            cap.release()
            out_safe.release()
            if out_mask is not None:
                out_mask.release()

        if processed_frames == 0:
            raise RuntimeError("입력 영상에서 처리된 프레임이 0개입니다.")

        self._verify_output_file(output_safe_path)
        if output_mask_path:
            self._verify_output_file(output_mask_path)

        elapsed = max(time.perf_counter() - start_time, 1e-6)

        return {
            "total_frames": processed_frames,
            "source_fps": round(src_fps, 2),
            "fps": round(effective_fps, 2),
            "processing_fps": round(processed_frames / elapsed, 2),
            "avg_psnr_db": round(float(np.mean(psnr_list)) if psnr_list else 0.0, 2),
            "detected_event_count": len(risk_events),
            "sample_events": risk_events[:10],
            "weights_loaded": self.weights_loaded,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI-7 Video Inference Demo")
    parser.add_argument("--input", type=str, required=True, help="입력 영상 경로")
    parser.add_argument("--output-safe", type=str, default="outputs/safe.mp4", help="보정 영상 출력 경로 (.mp4)")
    parser.add_argument("--output-mask", type=str, default="outputs/mask.mp4", help="마스크 오버레이 출력 경로 (.mp4)")
    parser.add_argument("--weights", type=str, default="lumi7_best.pth", help="학습된 모델 가중치 경로")
    parser.add_argument("--target-fps", type=float, default=None, help="고정 FPS 재표본화 옵션 (미지정 시 원본 FPS 사용)")
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
