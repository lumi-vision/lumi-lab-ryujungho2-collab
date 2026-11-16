"""영상 확인 전용 플레이어 — 판정 없이 보기만 한다 (`LUMI_영상보기.bat`).

    python scripts/view_videos.py [폴더] [--port 8770]

- 폴더 안(하위 폴더 포함)의 npz·mp4·webm 을 목록으로 보여 주고, 누르면 바로 재생한다.
- npz(생성 결과)는 처음 열 때 mp4 로 바꿔 옆에 저장한다(<이름>.mp4, 한 개에 1~2초). 원본(clean)이 들어 있으면
  왼쪽 원본 · 오른쪽 생성본으로 나란히 보여 준다. 판정기는 돌리지 않는다.
- 폴더를 비우면 가장 최근의 data/playlist/review/<시각>/ 을 연다. 페이지에서 다른 폴더 경로를 넣을 수 있다.
- 127.0.0.1 에서만 열린다. 고른 폴더 밖의 파일은 보내지 않는다.
⚠️ 광과민성 위험 자극이다. 이 PC 에서만, 밝기를 낮춰 짧게 본다. 공유·녹화하지 않는다.
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

EXT = {".npz", ".mp4", ".webm"}
lock = threading.Lock()


def default_folder() -> Path:
    rv = ROOT / "data" / "playlist" / "review"
    subs = sorted((p for p in rv.iterdir() if p.is_dir()), reverse=True) if rv.exists() else []
    return subs[0] if subs else ROOT / "data" / "playlist" / "generated"


def list_items(folder: Path) -> list[dict]:
    out = []
    seen_mp4_from_npz = set()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT and not p.name.endswith(".alpha.npz"))
    for p in files:
        if p.suffix.lower() == ".npz":
            seen_mp4_from_npz.add(p.with_suffix(".mp4"))
    for p in files:
        if p in seen_mp4_from_npz or p.name.endswith(".tmp.mp4"):
            continue                                   # npz 에서 만든 mp4 는 npz 항목으로 한 번만 보인다
        rel = p.relative_to(folder).as_posix()
        out.append({"rel": rel, "group": str(Path(rel).parent).replace("\\", "/"), "name": p.name,
                    "kind": "npz" if p.suffix.lower() == ".npz" else "video",
                    "ready": p.suffix.lower() != ".npz" or p.with_suffix(".mp4").exists()})
    return out


def ensure_mp4(npz: Path) -> Path:
    from npz_to_mp4 import frames_of
    from make_playlist import write_mp4
    mp4 = npz.with_suffix(".mp4")
    with lock:
        if not mp4.exists() or mp4.stat().st_mtime < npz.stat().st_mtime:
            fr, fps = frames_of(npz)
            write_mp4(mp4, iter(fr), fps)
    return mp4


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUMI 영상 보기</title><style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddd;--sel:#e6eefc;--warn:#fff4e5;--accent:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#e8e8e6;--muted:#9a9a96;--line:#333;--sel:#253250;--warn:#3a2c16;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,"Malgun Gothic",sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:8px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header input{flex:1;min-width:240px;padding:5px 8px;background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:6px}
button,select{padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.warn{background:var(--warn);padding:6px 12px;font-size:13px}
main{flex:1;display:flex;min-height:0}aside{width:300px;overflow:auto;border-right:1px solid var(--line)}
aside .g{padding:8px 12px 2px;color:var(--muted);font-size:12px}aside .it{padding:5px 12px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
aside .it:hover{background:var(--sel)}aside .it.cur{background:var(--sel);font-weight:600}aside .it .s{color:var(--muted);font-size:12px}
section{flex:1;display:flex;flex-direction:column;min-width:0;padding:10px;gap:8px}
video{width:100%;flex:1;min-height:0;background:#000;border-radius:6px}
.bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.muted{color:var(--muted)}
@media (max-width:700px){main{flex-direction:column}aside{width:auto;max-height:35vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<header><b>LUMI 영상 보기</b><input id="folder" placeholder="폴더 경로 (Enter)"><button id="open">열기</button><span id="status" class="muted"></span></header>
<div class="warn">⚠️ 광과민성 위험 자극입니다. 이 PC 에서만, 화면 밝기를 낮추고 짧게 보세요. 판정은 하지 않습니다 — 보기 전용.</div>
<main><aside id="list"></aside><section>
<div id="title" class="muted">왼쪽에서 영상을 고르세요 (↑/↓ 이전·다음, 스페이스 재생/멈춤, ←/→ 한 프레임)</div>
<video id="v" controls muted playsinline></video>
<div class="bar">속도 <button data-r="0.25">0.25×</button><button data-r="0.5">0.5×</button><button data-r="1">1×</button>
<button id="loop">반복</button><button id="auto">끝나면 다음</button><button id="prev">◀ 이전</button><button id="next">다음 ▶</button>
<button id="fb">−1 프레임</button><button id="ff">+1 프레임</button><span id="time" class="muted"></span></div>
</section></main>
<script>
const $=id=>document.getElementById(id), v=$("v"); let items=[], cur=-1, rate=+(localStorage.getItem("vv_rate")||1), loop=localStorage.getItem("vv_loop")!=="0", auto=localStorage.getItem("vv_auto")==="1";
const FPS=24;
function sync(){document.querySelectorAll("[data-r]").forEach(b=>b.classList.toggle("on",+b.dataset.r===rate));$("loop").classList.toggle("on",loop);$("auto").classList.toggle("on",auto);v.playbackRate=rate;v.loop=loop&&!auto}
async function load(folder){$("status").textContent="읽는 중…";const r=await fetch("/api/list?folder="+encodeURIComponent(folder||""));const j=await r.json();
 if(j.error){$("status").textContent=j.error;return} $("folder").value=j.folder;items=j.items;$("status").textContent=`${items.length}개`;render();if(items.length)select(0)}
function render(){const L=$("list");L.innerHTML="";let g=null;items.forEach((it,i)=>{if(it.group!==g){g=it.group;const d=document.createElement("div");d.className="g";d.textContent=g==="."?"(폴더)":g;L.appendChild(d)}
 const d=document.createElement("div");d.className="it"+(i===cur?" cur":"");d.innerHTML=`${it.name} <span class="s">${it.kind==="npz"&&!it.ready?"· 변환 필요":""}</span>`;d.onclick=()=>select(i);L.appendChild(d)})}
async function select(i){if(i<0||i>=items.length)return;cur=i;render();const it=items[i];$("title").textContent=it.rel+(it.ready?"":" — mp4 로 바꾸는 중…");
 const r=await fetch("/api/prepare?rel="+encodeURIComponent(it.rel));const j=await r.json();if(j.error){$("title").textContent=it.rel+" — "+j.error;return}
 it.ready=true;$("title").textContent=it.rel+(it.kind==="npz"?" · 왼쪽 원본 / 오른쪽 생성본":"");v.src="/file?rel="+encodeURIComponent(j.rel)+"&t="+Date.now();sync();v.play().catch(()=>{});render();
 const el=document.querySelector("#list .cur");if(el)el.scrollIntoView({block:"nearest"})}
v.onended=()=>{if(auto&&cur<items.length-1)select(cur+1)};v.ontimeupdate=()=>{$("time").textContent=`${v.currentTime.toFixed(2)}초 · 프레임 ${Math.floor(v.currentTime*FPS+1e-4)}`};
const step=d=>{v.pause();v.currentTime=Math.max(0,(Math.floor(v.currentTime*FPS+1e-4)+d)/FPS)};
document.querySelectorAll("[data-r]").forEach(b=>b.onclick=()=>{rate=+b.dataset.r;localStorage.setItem("vv_rate",rate);sync()});
$("loop").onclick=()=>{loop=!loop;localStorage.setItem("vv_loop",loop?"1":"0");sync()};$("auto").onclick=()=>{auto=!auto;localStorage.setItem("vv_auto",auto?"1":"0");sync()};
$("prev").onclick=()=>select(cur-1);$("next").onclick=()=>select(cur+1);$("fb").onclick=()=>step(-1);$("ff").onclick=()=>step(1);
$("open").onclick=()=>load($("folder").value.trim());$("folder").onkeydown=e=>{if(e.key==="Enter")load($("folder").value.trim())};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;if(e.key==="ArrowDown"){e.preventDefault();select(cur+1)}else if(e.key==="ArrowUp"){e.preventDefault();select(cur-1)}
 else if(e.key==="ArrowRight"){e.preventDefault();step(1)}else if(e.key==="ArrowLeft"){e.preventDefault();step(-1)}else if(e.key===" "){e.preventDefault();v.paused?v.play():v.pause()}});
try{localStorage}catch(e){} sync(); load("");
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    state = {"folder": Path(args.folder).resolve() if args.folder else default_folder()}
    hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj, code=200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def inside(self, rel: str) -> Path:
            p = (state["folder"] / rel).resolve()
            if not p.is_relative_to(state["folder"]) or not p.is_file():
                raise ValueError("폴더 밖이거나 없는 파일입니다")
            return p

        def do_GET(self):
            if self.headers.get("Host", "") not in hosts:
                return self.js({"error": "허용되지 않은 Host"}, 403)
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                if u.path == "/api/list":
                    if q.get("folder"):
                        f = Path(q["folder"].strip().strip('"')).expanduser().resolve()
                        if not f.is_dir():
                            return self.js({"error": f"폴더가 없습니다: {f}"})
                        state["folder"] = f
                    return self.js({"folder": str(state["folder"]), "items": list_items(state["folder"])})
                if u.path == "/api/prepare":
                    p = self.inside(q["rel"])
                    if p.suffix.lower() == ".npz":
                        p = ensure_mp4(p)
                    return self.js({"rel": p.relative_to(state["folder"]).as_posix()})
                if u.path == "/file":
                    return self.file(self.inside(q["rel"]))
            except (ValueError, KeyError, OSError) as e:
                return self.js({"error": f"{type(e).__name__}: {e}"}, 400)
            self.js({"error": "없음"}, 404)

        def file(self, p: Path):
            size = p.stat().st_size
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].partition("-")
                start, end = (int(a), int(b) if b else size - 1) if a else (max(0, size - int(b)), size - 1)
                end = min(end, size - 1)
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", "video/webm" if p.suffix.lower() == ".webm" else "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(p, "rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"영상 보기: {url}  (폴더: {state['folder']})  — 이 창을 닫으면 꺼집니다", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
