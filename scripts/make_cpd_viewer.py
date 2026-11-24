"""cpd 대역 실험 확인용 로컬 뷰어 — data/viewer/cpd/index.html (깃·공개 금지: South 재배포 불가).

- 자동 재생 없음: 초당 4 장 정지 프레임을 슬라이더·←/→ 로 한 장씩 넘긴다(광과민 안전 원칙).
- 칸 덮개: A(1–10, 회색) · B(1–8 Fisher, 파랑) · C(하한 없음 BT.1702, 주황) — 각 판정의 근거 프레임에서 고른 칸.
- 사람 확인: 클립마다 '규격상 실제 줄무늬(>5쌍·ΔL≥20·면적)' 예/아니오/모호 + 메모 → 브라우저에 저장, [내보내기]로 JSON.

    python scripts/make_cpd_viewer.py   →  data/viewer/cpd/index.html 을 브라우저로 열기
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "data" / "viewer" / "cpd"

HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>cpd 대역 확인</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#ddd;--a:#8a8a8a;--b:#2f6fd6;--c:#e07b00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9a94;--card:#202020;--line:#333}}
body{margin:0;font:14px/1.45 system-ui,"Malgun Gothic",sans-serif;background:var(--bg);color:var(--fg)}
.wrap{display:grid;grid-template-columns:300px 1fr;height:100vh}
aside{border-right:1px solid var(--line);overflow:auto;padding:10px}
main{overflow:auto;padding:14px}
.it{padding:6px 8px;border-radius:6px;cursor:pointer;font-size:12.5px;border:1px solid transparent}
.it:hover{border-color:var(--line)} .it.on{background:var(--card);border-color:var(--line)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:2px;border:1px solid var(--mut)}
.y{background:currentColor}
.lab{font-size:11px;color:var(--mut)}
canvas{max-width:100%;background:#000;border-radius:6px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}
table{border-collapse:collapse;font-size:13px} td,th{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:4px}
input[type=range]{width:420px;max-width:70vw}
textarea{width:100%;max-width:640px;height:48px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
button{padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.note{color:var(--mut);font-size:12px}
@media (max-width:760px){.wrap{grid-template-columns:1fr;height:auto}aside{max-height:35vh}}
</style></head><body><div class="wrap">
<aside><b>클립</b> <span class="note" id="cnt"></span>
<div class="row"><select id="flt"><option value="all">전체</option><option value="diff">판정 갈림만</option><option value="daily">일상만</option><option value="synth">합성 위험만</option><option value="todo">확인 안 한 것</option></select>
<button id="exp">내보내기</button></div><div id="list"></div></aside>
<main><div class="note">⚠️ 자동 재생 없음 — 정지 프레임을 한 장씩 넘깁니다(←/→). 칸 덮개는 각 판정의 근거 프레임 기준입니다.</div>
<h3 id="ttl" style="margin:8px 0"></h3><div class="note" id="why"></div>
<div class="row"><label><input type="checkbox" id="oA">A 1–10</label><label><input type="checkbox" id="oB" checked><span class="sw" style="background:var(--b)"></span>B 1–8 (Fisher)</label>
<label><input type="checkbox" id="oC" checked><span class="sw" style="background:var(--c)"></span>C 하한 없음 (BT.1702)</label></div>
<canvas id="cv"></canvas>
<div class="row"><button id="prev">◀</button><input type="range" id="sl" min="0" value="0"><button id="next">▶</button><span id="tt" class="note"></span>
<button id="toB">B 근거 프레임</button><button id="toC">C 근거 프레임</button></div>
<table id="tb"></table>
<h4 style="margin:14px 0 4px">사람 확인 — 규격상 실제 줄무늬인가? (>5쌍 · 밝기 차 ≥ 20 cd/m² · 면적)</h4>
<div class="row"><label><input type="radio" name="hv" value="yes">예, 실제 줄무늬</label><label><input type="radio" name="hv" value="no">아니오</label><label><input type="radio" name="hv" value="unsure">모호</label></div>
<textarea id="memo" placeholder="메모 (예: 블라인드, 창살, 바큇살 바깥쪽 …)"></textarea></main></div>
<script>
const D=__DATA__;let cur=0,fi=0;const K='lumi_cpd_check_v1';
let S={};try{S=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(K,JSON.stringify(S))}catch(e){}};
const cv=document.getElementById('cv'),cx=cv.getContext('2d');const img=new Image();
const col={A:getComputedStyle(document.documentElement).getPropertyValue('--a'),B:getComputedStyle(document.documentElement).getPropertyValue('--b'),C:getComputedStyle(document.documentElement).getPropertyValue('--c')};
function grp(d){return d.id.startsWith('south/')||d.id.startsWith('bg/')?'daily':'synth'}
function diff(d){const v=Object.values(d.res).map(r=>r.danger);return new Set(v).size>1}
function list(){const f=document.getElementById('flt').value;const L=document.getElementById('list');L.innerHTML='';let n=0;
D.forEach((d,i)=>{if(f==='diff'&&!diff(d))return;if((f==='daily'||f==='synth')&&grp(d)!==f)return;if(f==='todo'&&S[d.id]&&S[d.id].v)return;n++;
const e=document.createElement('div');e.className='it'+(i===cur?' on':'');
e.innerHTML=['A','B','C'].map(v=>`<span class="dot ${d.res[v].danger?'y':''}" style="color:${col[v]}"></span>`).join('')+' '+d.id+(S[d.id]&&S[d.id].v?` <span class="lab">✓${S[d.id].v}</span>`:'');
e.onclick=()=>{cur=i;fi=0;show();list()};L.appendChild(e)});document.getElementById('cnt').textContent=n+' 개'}
function show(){const d=D[cur];document.getElementById('ttl').textContent=d.id;document.getElementById('why').textContent=d.why;
const sl=document.getElementById('sl');sl.max=d.frames.length-1;sl.value=fi;
img.onload=draw;img.src=d.dir+'/f'+String(d.frames[fi]).padStart(4,'0')+'.jpg';
document.getElementById('tt').textContent=`${(d.frames[fi]/d.fps).toFixed(2)} 초 (${fi+1}/${d.frames.length})`;
const t=document.getElementById('tb');t.innerHTML='<tr><th>변형</th><th>판정</th><th>근거 시각</th><th>측정 cpd</th><th>쌍</th><th>면적</th><th>칸</th></tr>'+
['A','B','C'].map(v=>{const r=d.res[v];return `<tr><td><span class="sw" style="background:${col[v]}"></span>${v}</td><td>${r.danger?'위험':'안전'}${r.p2?' (변화)':''}</td><td>${r.t??'-'}</td><td>${r.cpd??'-'}</td><td>${r.pairs??'-'}</td><td>${r.area!=null?(100*r.area).toFixed(0)+'%':'-'}</td><td>${r.cells.length}</td></tr>`}).join('');
const s=S[d.id]||{};document.querySelectorAll('input[name=hv]').forEach(x=>x.checked=x.value===s.v);document.getElementById('memo').value=s.m||''}
function draw(){const d=D[cur];cv.width=img.naturalWidth*Math.max(1,Math.round(640/img.naturalWidth));cv.height=img.naturalHeight*cv.width/img.naturalWidth;
cx.imageSmoothingEnabled=false;cx.drawImage(img,0,0,cv.width,cv.height);
['A','B','C'].forEach(v=>{if(!document.getElementById('o'+v).checked)return;const r=d.res[v];const [R,Cn]=r.shape;const ch=cv.height/R,cw=cv.width/Cn;
cx.strokeStyle=col[v];cx.lineWidth=v==='C'?2:1.5;cx.setLineDash(v==='B'?[4,3]:[]);r.cells.forEach(([y,x])=>cx.strokeRect(x*cw+1,y*ch+1,cw-2,ch-2))});cx.setLineDash([])}
function go(k){const d=D[cur];fi=Math.max(0,Math.min(d.frames.length-1,fi+k));show()}
function jump(v){const d=D[cur],t=d.res[v].t;if(t==null)return;let b=0;d.frames.forEach((f,i)=>{if(Math.abs(f/d.fps-t)<Math.abs(d.frames[b]/d.fps-t))b=i});fi=b;show()}
document.getElementById('sl').oninput=e=>{fi=+e.target.value;show()};
document.getElementById('prev').onclick=()=>go(-1);document.getElementById('next').onclick=()=>go(1);
document.getElementById('toB').onclick=()=>jump('B');document.getElementById('toC').onclick=()=>jump('C');
['oA','oB','oC'].forEach(i=>document.getElementById(i).onchange=draw);document.getElementById('flt').onchange=list;
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)});
document.querySelectorAll('input[name=hv]').forEach(x=>x.onchange=()=>{const id=D[cur].id;S[id]=Object.assign(S[id]||{},{v:x.value});save();list()});
document.getElementById('memo').oninput=e=>{const id=D[cur].id;S[id]=Object.assign(S[id]||{},{m:e.target.value});save()};
document.getElementById('exp').onclick=()=>{const b=new Blob([JSON.stringify(S,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='cpd_human_check.json';a.click()};
list();if(D.length)show();
</script></body></html>"""


def main():
    items = []
    for m in sorted(V.glob("*/meta.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        d["dir"] = m.parent.name
        items.append(d)
    items.sort(key=lambda d: (not (len({r["danger"] for r in d["res"].values()}) > 1), d["id"]))
    (V / "index.html").write_text(HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False)), encoding="utf-8")
    print(f"{len(items)} 클립 → {V / 'index.html'}")


if __name__ == "__main__":
    main()
