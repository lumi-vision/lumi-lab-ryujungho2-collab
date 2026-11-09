"""적색 판정 정식 검증: pse-test-media video_creation 의 적색 세트 4개(영상 96개)를 원본 1920×1080 으로 재현해
표준별 기대값(expected_result)과 비교한다.

⚠️ 프레임은 메모리에서만 만들고 화면에 띄우거나 파일로 저장하지 않는다 (광과민성 위험 자극).
재현 코드는 scripts/validate_rules_pse.py 의 build_frames (pse-test-media generate_frames/generate_video 와 같은 계산,
첫 프레임 10장 반복)를 그대로 쓴다.

판정: 우리 = analyze_red 위험 OR analyze(휘도) 위험. 정답 = expected_result["itu_r1702_4"] (휘도 검증과 같은 키),
참고로 wcag2_2 기대값과의 일치도 함께 적는다 (WCAG 의 면적은 341×256 창 기준이라 우리와 정의가 다름).

사용: python scripts\\validate_red_sets.py
"""
import importlib.util
import json
import time
from pathlib import Path

from lumi_synth import measure

ROOT = Path(__file__).resolve().parents[1]
PSE = ROOT.parent / "tools/pse-test-media"
SETS = ["broadcast_30fps_red01", "broadcast_30fps_red02", "trace24_30fps_red01", "trace24_30fps_red02"]
OUT = ROOT / "checks/red/sets.md"
PADDING = 10

spec = importlib.util.spec_from_file_location("vr", ROOT / "scripts/validate_rules_pse.py")
vr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vr)


def norm(x):
    return None if x is None else ("위험" if str(x).lower().startswith("fail") else "안전")


def main() -> int:
    rows = []
    t0 = time.time()
    for s in SETS:
        for jp in sorted((PSE / "video_creation" / s).glob("*.json")):
            exp = json.loads(jp.read_text())["expected_result"]
            frames, fps = vr.build_frames(jp, PADDING)
            red = measure.analyze_red(frames, fps)
            lum = measure.analyze(frames, fps)
            ours = "위험" if (red["dangerous"] or lum["dangerous"]) else "안전"
            rows.append({"set": s, "name": jp.stem, "itu": norm(exp.get("itu_r1702_4")), "wcag": norm(exp.get("wcag2_2")),
                         "ours": ours, "red": red["dangerous"], "lum": lum["dangerous"],
                         "red_flashes": red["max_flashes"], "red_area": red["peak_area"], "duv": red["max_delta_uv"]})
            print(s, jp.stem, "기대(ITU)", rows[-1]["itu"], "우리", ours, f"적색 {red['max_flashes']:g}번 면적 {red['peak_area']:.1%}", flush=True)
            del frames
    sec = time.time() - t0

    def summary(key):
        r = [x for x in rows if x[key] is not None]
        tp = sum(x[key] == "위험" and x["ours"] == "위험" for x in r)
        fn = sum(x[key] == "위험" and x["ours"] == "안전" for x in r)
        fp = sum(x[key] == "안전" and x["ours"] == "위험" for x in r)
        tn = sum(x[key] == "안전" and x["ours"] == "안전" for x in r)
        return r, tp, fn, fp, tn

    r_itu, tp, fn, fp, tn = summary("itu")
    r_w, wtp, wfn, wfp, wtn = summary("wcag")
    lines = [
        "# 적색 판정 정식 검증: pse-test-media 적색 세트 4개 (영상 96개, 1920×1080)",
        "",
        f"> {time.strftime('%Y-%m-%d')} · 스크립트 `scripts/validate_red_sets.py` · 재현 `scripts/validate_rules_pse.py` `build_frames`(padding {PADDING})",
        f"> 판정: 우리 = 적색(`analyze_red`) 또는 휘도(`analyze`) 위험 · 정답 = `expected_result[\"itu_r1702_4\"]` · 판정 시간 {sec:.0f}초",
        "",
        f"**ITU 기대값과 {tp + tn}/{len(r_itu)} 일치 · 위험 놓침 {fn} · 과잉 검출 {fp}** (기대 위험 {tp + fn}개, 기대 안전 {fp + tn}개)",
        "",
        f"참고: WCAG 2.2 기대값과는 {wtp + wtn}/{len(r_w)} 일치 (놓침 {wfn}, 과잉 {wfp}). WCAG 면적은 341×256 창 기준이라 정의가 다름.",
        "",
        "| 세트 | 영상 수 | ITU 일치 | 놓침 | 과잉 |",
        "|---|---|---|---|---|",
    ]
    for s in SETS:
        sub = [x for x in rows if x["set"] == s and x["itu"]]
        lines.append(f"| {s} | {len(sub)} | {sum(x['itu'] == x['ours'] for x in sub)} | "
                     f"{sum(x['itu'] == '위험' and x['ours'] == '안전' for x in sub)} | "
                     f"{sum(x['itu'] == '안전' and x['ours'] == '위험' for x in sub)} |")
    bad = [x for x in r_itu if x["itu"] != x["ours"]]
    if bad:
        lines += ["", "## ITU 기대값과 다른 영상", "",
                  "| 세트 | 영상 | ITU 기대 | WCAG 기대 | 우리 | 적색 판정 | 휘도 판정 | 적색 1초 최대 | 적색 최대 면적 | 최대 Δu′v′ |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for x in bad:
            lines.append(f"| {x['set']} | `{x['name']}` | {x['itu']} | {x['wcag']} | {x['ours']} | "
                         f"{'위험' if x['red'] else '안전'} | {'위험' if x['lum'] else '안전'} | "
                         f"{x['red_flashes']:g} | {x['red_area']:.1%} | {x['duv']:.4f} |")
    lines += ["", "영상 이름: 앞 4글자 = 공간 패턴(a = 면적 미달, f = 면적 초과), 다음 = 시간 패턴(f/c/s/d, r = 순서 반전), 끝 m/s = multi/simple.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
