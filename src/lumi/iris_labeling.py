"""공개 검사기 IRIS(Electronic Arts, BSD-3-Clause)의 판정을 합성 데이터 라벨로 붙입니다.

IRIS 소스를 옮기지 않고 빌드된 예제 콘솔 앱을 subprocess로 호출합니다.
기록하는 값은 'IRIS 판정'이며 인증·임상 안전성 판정이 아닙니다(IRIS도 스스로 명시).

IRIS 프레임 결과 코드(include/iris/TotalFlashIncidents.h):
    휘도·적색 FlashResult: 0 Pass, 1 PassWithWarning, 2 ExtendedFail, 3 FlashFail
    패턴 PatternResult:   0 Pass, 1 Fail
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLASH_NAMES = {0: "pass", 1: "pass_with_warning", 2: "extended_fail", 3: "flash_fail"}
RESULT_COLUMNS = {"luminance": "LuminanceFrameResult", "red": "RedFrameResult",
                  "pattern": "PatternFrameResult"}
EXTRA_COLUMNS = ("LuminanceTransitions", "RedTransitions", "PatternDetectedLines")
WINDOW_RULES = ("last", "max")


class IrisError(RuntimeError):
    """IRIS 실행 또는 결과 해석 실패. 이 경우 통과로 취급하지 않습니다."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit(start: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


@dataclass(frozen=True)
class IrisFrames:
    """IRIS framedata.csv에서 읽은 프레임별 결과(0부터 시작하는 프레임 순서)."""

    luminance: np.ndarray
    red: np.ndarray
    pattern: np.ndarray
    luminance_transitions: np.ndarray
    red_transitions: np.ndarray
    pattern_lines: np.ndarray

    def __len__(self) -> int:
        return len(self.luminance)

    def to_json(self) -> dict:
        return {"luminance": self.luminance.tolist(), "red": self.red.tolist(),
                "pattern": self.pattern.tolist(),
                "luminance_transitions": self.luminance_transitions.tolist(),
                "red_transitions": self.red_transitions.tolist(),
                "pattern_lines": self.pattern_lines.tolist()}


def parse_framedata(text: str) -> IrisFrames:
    """IRIS framedata.csv 내용을 해석합니다.

    IRIS는 각 행 끝에 NUL 문자를 붙여 기록하므로 제거한 뒤 열 이름으로 읽습니다.
    """
    text = text.replace("\x00", "")
    reader = csv.DictReader(io.StringIO(text))
    required = list(RESULT_COLUMNS.values()) + list(EXTRA_COLUMNS) + ["Frame"]
    if reader.fieldnames is None or any(c not in reader.fieldnames for c in required):
        raise IrisError(f"framedata.csv 열이 예상과 다릅니다: {reader.fieldnames}")
    rows = [r for r in reader if r.get("Frame", "").strip()]
    if not rows:
        raise IrisError("framedata.csv에 프레임 행이 없습니다.")
    frames = [int(r["Frame"]) for r in rows]
    if frames != list(range(1, len(rows) + 1)):
        raise IrisError("framedata.csv 프레임 번호가 1부터 연속되지 않습니다.")

    def col(name: str) -> np.ndarray:
        return np.array([int(float(r[name])) for r in rows], dtype=np.int16)

    result = IrisFrames(col(RESULT_COLUMNS["luminance"]), col(RESULT_COLUMNS["red"]),
                        col(RESULT_COLUMNS["pattern"]), col("LuminanceTransitions"),
                        col("RedTransitions"), col("PatternDetectedLines"))
    if not (set(result.luminance) | set(result.red)) <= set(FLASH_NAMES):
        raise IrisError("휘도/적색 결과 코드가 0~3 범위를 벗어났습니다.")
    if not set(result.pattern) <= {0, 1}:
        raise IrisError("패턴 결과 코드가 0~1 범위를 벗어났습니다.")
    return result


def summarize(frames: IrisFrames) -> dict:
    """클립 단위 요약. 실패 프레임은 휘도/적색 2·3, 패턴 1인 프레임입니다."""
    return {
        "frames": len(frames),
        "luminance_max": int(frames.luminance.max()),
        "red_max": int(frames.red.max()),
        "pattern_max": int(frames.pattern.max()),
        "fail_frames": {
            "luminance": np.flatnonzero(frames.luminance >= 2).tolist(),
            "red": np.flatnonzero(frames.red >= 2).tolist(),
            "pattern": np.flatnonzero(frames.pattern >= 1).tolist(),
        },
    }


def standard_result(summary: dict) -> str:
    """요약을 'iris:...' 문자열로 바꿉니다. 여러 실패는 '+'로 잇습니다."""
    fails = []
    for kind in ("luminance", "red"):
        level = summary[f"{kind}_max"]
        if level >= 2:
            fails.append(f"{kind}_{FLASH_NAMES[level]}")
    if summary["pattern_max"] >= 1:
        fails.append("pattern_fail")
    if fails:
        return "iris:" + "+".join(fails)
    if max(summary["luminance_max"], summary["red_max"]) == 1:
        return "iris:pass_with_warning"
    return "iris:pass"


def window_label(frames: IrisFrames, start: int, length: int, rule: str) -> dict:
    """[start, start+length) 학습 창의 라벨.

    - "last": 창 마지막 프레임의 IRIS 결과(인과적, 실시간 탐지기와 같은 조건)
    - "max":  창 안의 최댓값(보수적)
    IRIS 프레임 결과는 해당 프레임까지의 직전 1초(연장은 5초)를 본 값이므로,
    창 앞쪽에 프레임이 부족해도 IRIS는 클립 전체 맥락으로 판정했다는 점에 유의합니다.
    """
    if rule not in WINDOW_RULES:
        raise ValueError(f"창 라벨 규칙은 {WINDOW_RULES} 중 하나여야 합니다.")
    if start < 0 or length < 1 or start + length > len(frames):
        raise ValueError("창 범위가 클립 길이를 벗어났습니다.")
    sl = slice(start, start + length)
    pick = (lambda a: int(a[sl][-1])) if rule == "last" else (lambda a: int(a[sl].max()))
    return {"luminance": pick(frames.luminance), "red": pick(frames.red),
            "pattern": pick(frames.pattern)}


class IrisRunner:
    """빌드된 IRIS 예제 앱(`-v <영상> -j 1 -p 1`)을 호출합니다.

    IRIS는 작업 디렉터리의 appsettings.json을 읽고 Results/<파일명>/에 결과를 씁니다.
    실행 파일 옆의 appsettings.json을 임시 작업 디렉터리로 복사해 실행하며,
    설정값은 바꾸지 않고 SHA-256만 기록합니다.
    """

    def __init__(self, exe: Path | str | None = None, *, timeout: float = 600.0) -> None:
        exe = exe or os.environ.get("LUMI_IRIS_EXE")
        if not exe:
            raise IrisError("IRIS 실행 파일 경로가 없습니다. --iris-exe 또는 LUMI_IRIS_EXE를 지정하세요.")
        self.exe = Path(exe).resolve()
        if not self.exe.is_file():
            raise IrisError(f"IRIS 실행 파일이 없습니다: {self.exe}")
        self.settings = self.exe.parent / "appsettings.json"
        if not self.settings.is_file():
            raise IrisError(f"appsettings.json이 실행 파일 옆에 없습니다: {self.settings}")
        self.timeout = timeout
        self.settings_sha256 = sha256_file(self.settings)
        self.version = os.environ.get("LUMI_IRIS_VERSION") or _git_commit(self.exe.parent) or "unknown"

    def run(self, video: Path) -> tuple[IrisFrames, str]:
        """영상 1개를 검사해 (프레임 결과, 원본 framedata.csv 텍스트)를 반환합니다."""
        video = Path(video).resolve()
        if not video.is_file():
            raise IrisError(f"영상이 없습니다: {video}")
        with tempfile.TemporaryDirectory(prefix="lumi_iris_") as tmp:
            work = Path(tmp)
            shutil.copy2(self.settings, work / "appsettings.json")
            try:
                proc = subprocess.run([str(self.exe), "-v", str(video), "-j", "1", "-p", "1"],
                                      cwd=work, capture_output=True, text=True,
                                      errors="replace", timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise IrisError(f"IRIS 시간 초과({self.timeout}s): {video.name}") from exc
            csv_path = work / "Results" / video.name / "framedata.csv"
            if proc.returncode != 0 or not csv_path.is_file():
                tail = (proc.stdout + proc.stderr)[-800:]
                raise IrisError(f"IRIS 실패(code {proc.returncode}): {video.name}\n{tail}")
            raw = csv_path.read_text(encoding="utf-8", errors="replace")
        return parse_framedata(raw), raw


def label_row(row: dict, root: Path, runner: IrisRunner) -> dict:
    """manifest 행 하나에 IRIS 판정을 붙입니다. 실패하면 iris_error로 남깁니다."""
    row = dict(row)
    video_rel = row.get("video")
    try:
        if not video_rel:
            raise IrisError("행에 무손실 영상(video)이 없습니다. --export-video로 다시 생성하세요.")
        frames, raw = runner.run(root / video_rel)
        if len(frames) != row["frames"]:
            raise IrisError(f"프레임 수 불일치: manifest {row['frames']}, IRIS {len(frames)}")
        stem = Path(video_rel).with_suffix("")
        frames_rel = f"{stem.as_posix()}.iris.json"
        (root / f"{stem.as_posix()}.iris.csv").write_text(raw, encoding="utf-8")
        (root / frames_rel).write_text(json.dumps(frames.to_json()), encoding="utf-8")
        summary = summarize(frames)
        row["iris"] = {"version": runner.version, "settings_sha256": runner.settings_sha256,
                       "video": video_rel, "frames_json": frames_rel, **summary}
        row["risk_verified"] = True  # "IRIS로 검사했다"는 뜻. 위험/안전 판정이 아님.
        row["standard_result"] = standard_result(summary)
    except IrisError as exc:
        row["iris"] = {"version": runner.version, "settings_sha256": runner.settings_sha256,
                       "error": str(exc)[:2000]}
        row["risk_verified"] = False
        row["standard_result"] = "iris_error"
    return row


def load_frames(root: Path, row: dict) -> IrisFrames:
    """label_row가 저장한 프레임별 결과를 다시 읽습니다."""
    data = json.loads((Path(root) / row["iris"]["frames_json"]).read_text(encoding="utf-8"))
    return IrisFrames(*(np.asarray(data[k], dtype=np.int16) for k in (
        "luminance", "red", "pattern", "luminance_transitions", "red_transitions",
        "pattern_lines")))
