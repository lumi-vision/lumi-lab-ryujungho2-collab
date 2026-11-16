"""Windows 실행 파일이 사용하는 서버 시작 진입점. 로그는 부모 프로세스와 독립적으로 유지한다."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port out of range")
    root = Path(__file__).resolve().parents[1]
    logs = root / "data/launcher"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with (logs / f"server-{args.port}.log").open("ab", buffering=0) as log:
        proc = subprocess.Popen([sys.executable, "-u", str(root / "scripts/make_playlist.py"),
                                 "--serve", "--no-browser", "--port", str(args.port)], cwd=root,
                                stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    (logs / f"server-{args.port}.pid").write_text(str(proc.pid), encoding="ascii")


if __name__ == "__main__":
    main()
