"""LUMI-7 실시간 스트리밍 추론 및 시연 영상 생성 모듈.

학습된 가중치(lumi7_best.pth)를 로드해 입력 영상을 프레임 단위로 탐지·보정하고,
안전 보정 영상(safe mp4)과 위험 히트맵 오버레이 영상(mask mp4) 및 진단 리포트를 생성한다.
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
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor = CausalAuxExtractor().to(self.device).eval()
        self.model = LUMI7(aux_channels=9).to(self.device).eval()

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True
        else:
            self.weights_loaded = False

    @torch.no_grad()
    def process_video(
        self,
        input_video_path: str,
        output_safe_path: str,
        output_mask_path: Optional[str] = None,
        risk_threshold: float = 0.5,
    ) -> dict:
        """MP4 파일을 프레임 단위로 처리해 보정 영상과 진단 통계를 반환한다."""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"입력 영상을 열 수 없습니다: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        Path(output_safe_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_safe = cv2.VideoWriter(output_safe_path, fourcc, fps, (width, height))

        out_mask = None
        if output_mask_path:
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            out_mask = cv2.VideoWriter(output_mask_path, fourcc, fps, (width, height))

        aux_state = None
        model_state = None

        frame_idx = 0
        psnr_list = []
        risk_events = []
        start_time = time.perf_counter()

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

            # 1. 인과적 특징 추출 -> 2. 탐지 및 국소 보정
            aux, aux_state = self.extractor.step(rgb_tensor, aux_state)
            out = self.model.step(rgb_tensor, aux, model_state)
            model_state = out["state"]

            # 보정 프레임 변환
            safe_rgb = (
                (out["corrected"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            safe_bgr = cv2.cvtColor(safe_rgb, cv2.COLOR_RGB2BGR)
            out_safe.write(safe_bgr)

            # 탐지 마스크 분석
            mask_np = out["mask"].squeeze().cpu().numpy()
            active_ratio = float((mask_np > risk_threshold).mean())

            if active_ratio > 0.05:
                sec = frame_idx / fps
                risk_events.append(
                    {
                        "frame": frame_idx,
                        "timestamp": f"{int(sec // 60):02d}:{sec % 60:05.2f}",
                        "affected_area_ratio": round(active_ratio, 4),
                    }
                )

            # PSNR 계산 (원본 대비 화질 보존도)
            mse = float(np.mean((frame_rgb.astype(np.float32) - safe_rgb.astype(np.float32)) ** 2))
            psnr = 20.0 * np.log10(255.0 / np.sqrt(mse)) if mse > 1e-6 else 50.0
            psnr_list.append(psnr)

            if out_mask is not None:
                mask_u8 = (mask_np * 255.0).clip(0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame_bgr, 0.7, heatmap, 0.3, 0)
                out_mask.write(overlay)

            frame_idx += 1

        elapsed = max(time.perf_counter() - start_time, 1e-6)
        cap.release()
        out_safe.release()
        if out_mask is not None:
            out_mask.release()

        return {
            "total_frames": frame_idx,
            "fps": round(fps, 2),
            "processing_fps": round(frame_idx / elapsed, 2),
            "avg_psnr_db": round(float(np.mean(psnr_list)) if psnr_list else 0.0, 2),
            "detected_event_count": len(risk_events),
            "sample_events": risk_events[:10],
            "weights_loaded": self.weights_loaded,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI-7 Video Inference Demo")
    parser.add_argument("--input", type=str, required=True, help="입력 MP4 경로")
    parser.add_argument("--output-safe", type=str, default="outputs/safe.mp4", help="보정 영상 출력 경로")
    parser.add_argument("--output-mask", type=str, default="outputs/mask.mp4", help="마스크 오버레이 출력 경로")
    parser.add_argument("--weights", type=str, default="lumi7_best.pth", help="학습된 모델 가중치 경로")
    args = parser.parse_args()

    inferencer = LUMIInferencer(weights_path=args.weights)
    report = inferencer.process_video(
        input_video_path=args.input,
        output_safe_path=args.output_safe,
        output_mask_path=args.output_mask,
    )
    print("[INFERENCE COMPLETE]")
    print(report)


if __name__ == "__main__":
    main()
