"""South '안전' × 팀원 판 정지 패턴 840 — 이유 확인 뷰어 (로컬·비공개 전용, data/viewer/south_v1g/index.html).

- 보통 모드: 3 칸 그림(입력 | 팀원 판이 본 것 | G 가 본 것) + 두 판정기의 수치·G 가 멈춘 단계. 이유별·경로별 거르기.
- 블라인드 모드: 왼쪽 입력 프레임만 보이고 판정기 정보·순서가 가려짐(섞인 순서 고정) → 사람 판정용.
- 사람 판정: '규격상 정지 패턴인가 — 뚜렷한 줄무늬 5쌍 초과 · 화면 40 % 초과' 예/아니오/모호 + 메모 → 브라우저 저장, [내보내기] JSON.
정지 프레임만 사용(재생 없음).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "data" / "viewer" / "south_v1g"
REASON = {"G0": "줄무늬 칸 없음", "G1": "후보 영역 원면적 ≤ 40 %", "G2": "영역 > 40 % · 법선 경로에 6쌍 없음",
          "G3": "6쌍 경로는 있으나 규칙성 실패", "G4": "G 도 이 프레임은 패턴"}

HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>South 팀원 판 vs G</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#ddd;--v1:#c25e00;--g:#2f6fd6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9a94;--card:#202020;--line:#333}}
body{margin:0;font:14px/1.45 system-ui,"Malgun Gothic",sans-serif;background:var(--bg);color:var(--fg)}
.wrap{display:grid;grid-template-columns:320px 1fr;height:100vh}
aside{border-right:1px solid var(--line);overflow:auto;padding:10px}
main{overflow:auto;padding:16px}
.it{padding:5px 8px;border-radius:6px;cursor:pointer;font-size:12.5px;border:1px solid transparent}
.it:hover{border-color:var(--line)}.it.on{background:var(--card);border-color:var(--line)}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;background:var(--line);margin-left:4px}
.frame{overflow:hidden;border-radius:8px;background:#000}
.frame img{display:block;image-rendering:pixelated}
table{border-collapse:collapse;font-size:13px;margin-top:10px}td,th{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}
button,select{padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
textarea{width:100%;max-width:700px;height:46px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.note{color:var(--mut);font-size:12px}
.k{font-weight:700}
@media (max-width:760px){.wrap{grid-template-columns:1fr;height:auto}aside{max-height:35vh}}
</style></head><body><div class="wrap">
<aside><div class="row"><label><input type="checkbox" id="blind"> 블라인드 모드</label><button id="exp">내보내기</button></div>
<div class="row"><select id="fw"><option value="">G 단계: 전체</option>__OPTS__</select>
<select id="fs"><option value="">팀원 판 경로: 전체</option><option>직선 주사</option><option>고리·바큇살</option><option>FFT 대체</option></select>
<select id="fd"><option value="">판정: 전체</option><option value="todo">아직 안 봄</option><option value="done">본 것</option></select></div>
<div class="note" id="cnt"></div><div id="list"></div></aside>
<main><div class="note">정지 프레임만 · 라벨 '안전'은 PhotosensitivityPal(자동 검출기) 출력이지 사람 판정이 아님.</div>
<h3 id="ttl" style="margin:8px 0"></h3>
<div class="frame" id="fbox"><img id="im" alt=""></div>
<div id="info"></div>
<h4 style="margin:14px 0 4px">사람 판정 — 규격상 정지 패턴인가? (뚜렷한 줄무늬 5쌍 초과 · 화면 40 % 초과)</h4>
<div class="row"><label><input type="radio" name="hv" value="yes">예, 패턴</label><label><input type="radio" name="hv" value="no">아니오</label><label><input type="radio" name="hv" value="unsure">모호</label>
<button id="prev">◀ 이전</button><button id="next">다음 ▶</button></div>
<textarea id="memo" placeholder="메모 (예: 글자, 블라인드, 군중, 나뭇잎 …)"></textarea></main></div>
<script>
const D=__DATA__,RS=__REASON__;const K='lumi_south_v1g_v1';let S={};try{S=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const save=()=>{try{localStorage.setItem(K,JSON.stringify(S))}catch(e){}};
let order=D.map((d,i)=>i),cur=0,view=[];
function shuf(a){let s=20261126;a=a.slice();for(let i=a.length-1;i>0;i--){s=(s*1103515245+12345)%2147483648;const j=s%(i+1);[a[i],a[j]]=[a[j],a[i]]}return a}
function build(){const b=document.getElementById('blind').checked,fw=fw_.value,fs=fs_.value,fd=fd_.value;
let idx=b?shuf(order):order;view=idx.filter(i=>{const d=D[i];if(!b&&fw&&d.why!==fw)return false;if(!b&&fs&&d.src!==fs)return false;
const h=S[d.id]&&S[d.id].v;if(fd==='todo'&&h)return false;if(fd==='done'&&!h)return false;return true});
const L=document.getElementById('list');L.innerHTML='';view.forEach((i,k)=>{const d=D[i],e=document.createElement('div');e.className='it'+(k===cur?' on':'');
e.innerHTML=(b?('#'+(k+1)):d.id+'<span class="tag">'+d.why+'</span>')+(S[d.id]&&S[d.id].v?' ✓'+S[d.id].v:'');e.onclick=()=>{cur=k;show();build()};L.appendChild(e)});
document.getElementById('cnt').textContent=view.length+' 개';if(cur>=view.length)cur=0;show()}
function show(){if(!view.length)return;const d=D[view[cur]],b=document.getElementById('blind').checked;
document.getElementById('ttl').textContent=b?('#'+(cur+1)+' / '+view.length):d.id;const im=document.getElementById('im'),fb=document.getElementById('fbox');
im.src=d.img;im.onload=()=>{const s=Math.min(2.4,(document.querySelector('main').clientWidth-40)/(b?300:900));fb.style.width=(b?300:900)*s+'px';im.style.width=900*s+'px'};
document.getElementById('info').innerHTML=b?'':`<table><tr><th></th><th>팀원 판</th><th>G</th></tr>
<tr><td>판정(이 프레임)</td><td>정적 공간 패턴</td><td>${RS[d.why]}</td></tr>
<tr><td>면적</td><td>대비 지도 ${(100*d.fine).toFixed(0)} %</td><td>가장 큰 후보 영역 ${(100*d.raw).toFixed(1)} %</td></tr>
<tr><td>쌍</td><td>${d.pairs}쌍 (${d.src})</td><td>${d.tested}</td></tr>
<tr><td>규칙성(직선 구간에 G 규칙)</td><td colspan=2>${d.reg==null?'-':d.reg.toFixed(2)+(d.reg>0.25?' → 탈락':' → 통과')}</td></tr>
<tr><td>G 클립 판정</td><td colspan=2>${d.clip?'위험':'안전'} · BT.1702 축 이 프레임 ${d.bt?'패턴':'아님'}</td></tr></table>
<div class="note">그림: 왼쪽 입력 · 가운데 팀원 판(주황 = 대비 지도, 노랑 = 6쌍 이상 센 직선 구간) · 오른쪽 G(노란 칸 = 후보 칸, 초록/빨강 = 확인 성공/실패 경로)</div>`;
const s=S[d.id]||{};document.querySelectorAll('input[name=hv]').forEach(x=>x.checked=x.value===s.v);document.getElementById('memo').value=s.m||''}
const fw_=document.getElementById('fw'),fs_=document.getElementById('fs'),fd_=document.getElementById('fd');
[fw_,fs_,fd_,document.getElementById('blind')].forEach(e=>e.onchange=()=>{cur=0;build()});
document.querySelectorAll('input[name=hv]').forEach(x=>x.onchange=()=>{const id=D[view[cur]].id;S[id]=Object.assign(S[id]||{},{v:x.value});save();build()});
document.getElementById('memo').oninput=e=>{const id=D[view[cur]].id;S[id]=Object.assign(S[id]||{},{m:e.target.value});save()};
document.getElementById('prev').onclick=()=>{cur=Math.max(0,cur-1);show();build()};document.getElementById('next').onclick=()=>{cur=Math.min(view.length-1,cur+1);show();build()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowRight'){cur=Math.min(view.length-1,cur+1);show();build()}if(e.key==='ArrowLeft'){cur=Math.max(0,cur-1);show();build()}});
document.getElementById('exp').onclick=()=>{const b=new Blob([JSON.stringify(S,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='south_v1g_human.json';a.click()};
build();
</script></body></html>"""


def main():
    rows = [json.loads(l) for l in (ROOT / "data/judge/south_v1g.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    D = []
    for r in rows:
        if "error" in r:
            continue
        t = r["G"]["regions_tested"]
        tested = "; ".join(f"영역 {100 * q['raw']:.0f}%: 경로 {q['n']}개 중 6쌍 {q['ge6']} · 확인 {q['ok']}" for q in t) or "원면적 40 % 넘는 영역 없음 → 확인 생략"
        D.append({"id": r["id"], "img": f"{r['id']}.jpg", "why": r["G"]["why"], "src": r["v1"]["src"], "fine": r["v1"]["fine_area"],
                  "raw": r["G"]["max_raw"], "pairs": r["v1"]["pairs"], "reg": r["v1"]["line_reg_G"], "tested": tested,
                  "clip": r["G_clip"], "bt": r["G_bt1702_frame"]})
    opts = "".join(f'<option value="{k}">{k} {v}</option>' for k, v in REASON.items())
    html = HTML.replace("__DATA__", json.dumps(D, ensure_ascii=False)).replace("__REASON__", json.dumps(REASON, ensure_ascii=False)).replace("__OPTS__", opts)
    (V / "index.html").write_text(html, encoding="utf-8")
    print(len(D), "→", V / "index.html")


if __name__ == "__main__":
    main()
