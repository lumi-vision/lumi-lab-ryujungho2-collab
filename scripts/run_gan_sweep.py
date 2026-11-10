"""GAN 시드 반복 실험 실행기: GPU 학습은 하나씩 차례로, 끝난 실험의 평가(CPU 하드 판정)는 다음 학습과 동시에.

GPU 는 한 학습이 거의 다 쓰므로(768×432×48, 2026-11-10 측정) 학습을 동시에 둘 돌리지 않는다.
평가는 G 실행과 하드 판정(measure.analyze)이 CPU 라서 GPU 학습과 겹쳐도 서로 방해가 적다.

사용
    python scripts/run_gan_sweep.py                       # λ_div 0.05/0.5 × 시드 0·1·2, 500스텝 (HOME_NEXT_STEPS §6)
    python scripts/run_gan_sweep.py --steps 50 --seeds 0 --divs 0.5 --tag smoke   # 동작 확인
결과: checks/gan/<tag>_div<d>_s<s>.md (+ _rows.jsonl, train_log_*.jsonl), 체크포인트·샘플은 data/gan/ (Git 제외)
이미 체크포인트·점검표가 있는 실험은 건너뛴다(중단 후 다시 실행하면 이어서 진행).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--divs", default="0.05,0.5")
    ap.add_argument("--tag", default="u")
    ap.add_argument("--fp32", action="store_true", help="bf16 를 끄고 fp32 로 학습(bf16 영향 분리용, 약 2배 느림)")
    ap.add_argument("--eval-threads", type=int, default=6, help="평가 하나가 쓰는 CPU 스레드")
    ap.add_argument("--max-evals", type=int, default=2, help="동시에 돌릴 평가 수")
    args = ap.parse_args()

    logs = ROOT / "data" / "gan" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(float(d), int(s)) for s in args.seeds.split(",") for d in args.divs.split(",")]
    evals: list[tuple[str, subprocess.Popen]] = []
    env_eval = os.environ | {"OMP_NUM_THREADS": str(args.eval_threads), "MKL_NUM_THREADS": str(args.eval_threads)}

    def reap(block_until_below: int) -> None:
        while True:
            for name, p in list(evals):
                if p.poll() is not None:
                    print(f"[평가 끝] {name} 종료 코드 {p.returncode}", flush=True)
                    evals.remove((name, p))
            if len(evals) < block_until_below:
                return
            time.sleep(10)

    t0 = time.time()
    for d, s in jobs:
        name = f"{args.tag}_div{d}_s{s}"
        ck = ROOT / "data" / "gan" / f"{name}.pt"
        report = ROOT / "checks" / "gan" / f"{name}.md"
        if not ck.exists():
            cmd = [PY, "-u", str(ROOT / "scripts" / "train_gan.py"), "--device", "cuda", "--grad-ckpt",
                   "--steps", str(args.steps), "--batch-size", str(args.batch_size), "--micro-batch", str(args.micro_batch),
                   "--width", str(args.width), "--lambda-div", str(d), "--seed", str(s),
                   "--checkpoint", str(ck), "--log", str(ROOT / "checks" / "gan" / f"train_log_{name}.jsonl")]
            if not args.fp32:
                cmd.append("--amp")
            print(f"[학습 시작] {name} ({(time.time() - t0) / 60:.1f}분 경과)", flush=True)
            with open(logs / f"train_{name}.log", "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"[학습 끝] {name} 종료 코드 {rc} ({(time.time() - t0) / 60:.1f}분 경과)", flush=True)
            if rc != 0:
                print(f"  → 실패. 로그: {logs / f'train_{name}.log'}", flush=True)
                continue
        if report.exists():
            continue
        reap(args.max_evals)
        cmd = [PY, "-u", str(ROOT / "scripts" / "eval_gan.py"), "--checkpoint", str(ck),
               "--out", str(ROOT / "data" / "gan" / name), "--report", str(report)]
        f = open(logs / f"eval_{name}.log", "w", encoding="utf-8")
        evals.append((name, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env_eval)))
        print(f"[평가 시작] {name}", flush=True)
    reap(1)
    print(f"[전체 끝] {(time.time() - t0) / 60:.1f}분", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
