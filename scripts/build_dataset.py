"""명령행에서 LUMI의 기본 합성 데이터셋을 만듭니다."""
import argparse
from pathlib import Path

from lumi.dataset_synthesis import DatasetConfig, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="PSE 연구용 통제 합성 데이터셋 생성(미리보기 없음)")
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="원본 영상 경로(여러 번 지정 가능). 검증된 일반 영상만 사용하세요.")
    parser.add_argument("--output", type=Path, default=Path("datasets/oracle"))
    parser.add_argument("--samples", type=int, default=20, help="원본 영상당 합성 클립 수")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-seconds", type=int, default=60,
                        help="각 원본 영상에서 읽을 최대 길이(초), 메모리 사용량 제한")
    args = parser.parse_args()
    cfg = DatasetConfig(frames=args.frames, width=args.width, height=args.height,
                        fps=args.fps, seed=args.seed,
                        max_source_seconds=args.max_source_seconds)
    entries = build_dataset(args.source, args.output, samples_per_source=args.samples, cfg=cfg)
    count = {k: sum(v["type"] == k for v in entries) for k in ("normal", "luminance", "red", "pattern")}
    print(f"합성 완료: {len(entries)}개, 유형별 개수={count}")
    print(f"메타데이터: {args.output / 'manifest.jsonl'}")
    print("경고: 데이터의 실제 PSE 위험 여부는 아직 검증되지 않았습니다. 자동 미리보기는 제공하지 않습니다.")


if __name__ == "__main__":
    main()
