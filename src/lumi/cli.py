"""LUMI 1단계 명령행 인터페이스."""
from __future__ import annotations

import argparse
import json
import sys

import cv2

from lumi.analysis import scan_video
from lumi.player import preview
from lumi.video_io import VideoReader


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lumi", description="LUMI: PSE 연구용 영상 입출력 시제품")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "scan", "preview"):
        sub = commands.add_parser(name)
        sub.add_argument("input", help="영상 파일 경로(MP4, AVI, MOV 등)")
        sub.add_argument("--backend", choices=("auto", "pyav", "opencv"), default="auto")
        if name == "scan":
            sub.add_argument("--output", default="output/frames.jsonl")
            sub.add_argument("--start", type=float, default=0.0, help="시작 시각(초)")
            sub.add_argument("--end", type=float, default=None, help="종료 시각(초, 미포함)")
            sub.add_argument("--max-frames", type=int, default=None)
        elif name == "preview":
            sub.add_argument("--allow-preview", action="store_true", help="필터가 없는 영상 미리보기를 명시적으로 허용")
            sub.add_argument("--history", type=int, default=12, help="메모리에 저장할 최대 원본 크기 프레임 수")
    static = commands.add_parser("static", help="정지 이미지 1장의 규칙 패턴 스크리닝과 공간 특징 (화면에 표시하지 않음)")
    static.add_argument("input", help="이미지 파일 경로(PNG, JPG 등)")
    static.add_argument("--diagonal-inch", type=float, default=27.0, help="화면 대각선(인치)")
    static.add_argument("--distance-cm", type=float, default=60.0, help="시청 거리(cm)")
    static.add_argument("--mask-out", default=None, help="검출된 패턴 영역 마스크를 PNG로 저장")
    return p


def _static(args) -> None:
    import cv2
    import numpy as np
    from lumi.static_features import ViewingGeometry, analyze_image, load_image_rgb
    result = analyze_image(load_image_rgb(args.input),
                           geometry=ViewingGeometry(args.diagonal_inch, distance_cm=args.distance_cm))
    mask = result.pop("mask")
    if args.mask_out:
        ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not ok:
            raise RuntimeError("마스크 PNG 인코딩 실패")
        buf.tofile(args.mask_out)
        result["mask_png"] = args.mask_out
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            with VideoReader(args.input, args.backend) as reader:
                print(json.dumps(reader.info.as_dict(), indent=2, ensure_ascii=False))
        elif args.command == "scan":
            result = scan_video(args.input, args.output, args.backend,
                                args.start, args.end, args.max_frames)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "preview":
            preview(args.input, args.backend, args.allow_preview, args.history)
        elif args.command == "static":
            _static(args)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, PermissionError, cv2.error) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
