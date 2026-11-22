"""정상 단색 배열로 공통 평가 인터페이스의 동작을 확인한다."""
import argparse
import json
import numpy as np
from lumi_synth.baselines.comparison import infer


def main():
    """테스트용 계수만 사용하며 연구 성능 결과를 생성하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='동작 확인할 로컬 HazNet 체크포인트')
    args = parser.parse_args()
    frames = np.full((48, 32, 48, 3), 80, dtype=np.uint8)
    results = [infer(frames, 24, 'B0'), infer(frames, 24, 'B1', coefficient=1, intercept=-1)]
    if args.checkpoint:
        results.append(infer(frames, 24, 'B2', checkpoint=args.checkpoint))
    print(json.dumps({'purpose': '정상 단색 배열 인터페이스 테스트; 성능 평가 아님', 'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
