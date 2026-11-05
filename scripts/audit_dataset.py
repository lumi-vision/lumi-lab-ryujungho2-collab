"""데이터셋 정합성 감사를 수행합니다."""
import argparse
import json
from pathlib import Path

from lumi.dataset_audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 데이터 정합성 및 분할 누출 검사")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/oracle"))
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
