"""합성 데이터셋의 무손실 영상을 IRIS로 검사해 manifest에 판정을 기록합니다.

영상은 재생·미리보기하지 않고 IRIS 실행 파일에만 입력합니다.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from lumi.iris_labeling import IrisRunner, label_row


def main() -> None:
    parser = argparse.ArgumentParser(description="IRIS 판정으로 risk_verified/standard_result 채우기")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iris-exe", type=Path, default=None,
                        help="IRIS 예제 앱 경로(기본: 환경 변수 LUMI_IRIS_EXE)")
    parser.add_argument("--timeout", type=float, default=600.0, help="영상 1개당 제한 시간(초)")
    args = parser.parse_args()
    runner = IrisRunner(args.iris_exe, timeout=args.timeout)
    manifest = args.dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"IRIS {runner.version[:12]} · settings {runner.settings_sha256[:12]} · 영상 {len(rows)}개")
    labeled = []
    for i, row in enumerate(rows, 1):
        out = label_row(row, args.dataset, runner)
        labeled.append(out)
        (args.dataset / Path(out["file"]).with_suffix(".json")).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(rows)}] {out['id']} {out['type']:>9} → {out['standard_result']}")
    tmp = manifest.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labeled), encoding="utf-8")
    tmp.replace(manifest)

    table = Counter((r["type"], r["standard_result"]) for r in labeled)
    print("\n합성 유형(조건) × IRIS 판정")
    for (kind, result), n in sorted(table.items()):
        print(f"  {kind:>9}  {result:<45} {n}")
    errors = sum(r["standard_result"] == "iris_error" for r in labeled)
    if errors:
        print(f"경고: iris_error {errors}개는 학습에서 제외해야 합니다(통과로 취급 금지).")
    print("IRIS 판정은 인증·임상 안전성 판정이 아닙니다.")


if __name__ == "__main__":
    main()
