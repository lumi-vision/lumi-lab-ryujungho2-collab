# 데이터 대장 — 무엇을 어디서 받았고, 왜 받았으며, 어디에 있는가

> 기준일 2026-11-07 · 관리: GPU PC(`C:\dev\lumi-vision`). 이 문서에는 **데이터 자체를 넣지 않는다.** 데이터는 모두 저장소 밖 `_work`·`_ext`에 있고 git에 올리지 않는다.
> 새로 받거나 만들 때마다 이 대장에 한 줄을 추가한다. 지울 때는 지운 날짜를 적는다.

## 0. 공통 원칙

| 원칙 | 내용 |
|---|---|
| 저장 위치 | 외부 데이터·생성 데이터 `C:\dev\lumi-vision\_work\`, 외부 도구 `C:\dev\lumi-vision\_ext\`. 저장소 폴더에는 두지 않는다([COLLABORATION.md](COLLABORATION.md)) |
| 유해 자극 | 깜빡임·패턴이 든 영상은 **재생·미리보기하지 않는다.** IRIS 등 검사기에만 입력한다 |
| 공유 | 데이터 파일은 git·클라우드에 올리지 않는다. 팀원에게는 이 대장의 출처·명령을 주고 각자 받거나 다시 만든다 |
| 라이선스 | 라이선스가 없거나 불분명한 자료는 내부 연구용으로만 쓰고 재배포하지 않는다 |

## 1. 외부에서 받은 데이터

### X1. South, Saffo & Borkin (2021) 소셜미디어 GIF·영상 2,000개

| 항목 | 내용 |
|---|---|
| 논문 | L. South, D. Saffo, M. A. Borkin, "Detecting and Defending Against Seizure-Inducing GIFs in Social Media", *CHI 2021*, doi:10.1145/3411764.3445510 |
| 받은 곳 | OSF 프로젝트 https://osf.io/5a3dy/ (논문의 사전 등록·보충 자료). 사전 공개본은 https://osf.io/4kgu6/ |
| 받은 것 | ① 미디어 2,000개 = GIF 1,500 + MP4 500 (3.2 GB) ② 라벨 CSV `social-media-study.csv` ③ 인터뷰 질문지 PDF ④ 논문 사전 공개본 PDF |
| 내용 | Twitter·Tumblr·Tenor·GIPHY에서 각 500개를 무작위 수집한 실제 GIF·영상. CSV 열: `label, id, url, source, dangerous, flash, red, pattern` |
| 라벨 | 위험 163 / 안전 1,837. 깜빡임 94, 패턴 67, 적색 12. **사람 라벨이 아니라 저자 검출기(PhotosensitivityPal)의 판정**: WCAG 2.0 휘도(상대 휘도 변화 ≥ 10%, 어두운 쪽 < 0.8, 면적 > 25%)·적색 규칙 + Wilkins식 패턴 |
| 원 논문에서의 용도 | 소셜미디어의 발작 유발 GIF 유병률 추정(8.15%, 99% CI 6.65–9.85%)과 플랫폼 간 비교 |
| 라이선스 | **표기 없음.** 콘텐츠 저작권은 원 게시자. 내부 연구용으로만 사용, 재배포 금지 |
| 우리가 받은 이유 | (a) **실제 일반 영상**(안전 1,837개): Oracle 배경과 D의 "자연스러움" 학습용. 지금까지는 합성 질감 영상뿐이었다. (b) **실제 위험 영상**(163개): G가 흉내 낼 실제 분포의 보조 신호. (c) Oracle과 무관한 **외부 시험셋**(H1 평가). (d) 판정기(저자 검출기·IRIS·R) 간 차이를 실제 영상에서 측정 |
| 지금까지 한 일 | 전체를 IRIS로 판정, 저자 라벨과 비교([IRIS_LABELING_PLAN.md](IRIS_LABELING_PLAN.md) 5.3). 자료 특성: 가로 중앙값 480 px, 11 fps(위험 17 fps), 길이 중앙값 2.4초 |
| 위치 | 미디어 `_work\datasets\south2021\media\` · CSV·논문·받기 스크립트 `_work\refs\osf_south2021\` |
| 받은 날·방법 | 2026-11-07, OSF API(`download_osf.py`, 동시 2개 + 429 대기 재시도). 실패 0 |
| 보관 | **연구 종료 후 `datasets\south2021\` 전체 삭제**(사용자 결정 2026-11-07). CSV·논문 PDF는 그때 결정 |
| 주의 | 위험 163개 포함. 재생 금지. 어떤 파일이 위험인지는 CSV `dangerous` 열 |

### X2. pse-test-media (TRACE RERC) 시험 영상 생성기

| 항목 | 내용 |
|---|---|
| 논문 | J. B. Jordan, "Evaluating Conformance of Video Safety Tools for Photosensitive Epilepsy", *UAHCI 2025*, LNCS 15780:85–98, doi:10.1007/978-3-031-93848-1_7 (저자본 PMC12249941) |
| 받은 곳 | https://github.com/traceRERC/pse-test-media (커밋 `ef78542`, 2026-09-09) |
| 받은 것 | 생성기 코드(Python), 공간 패턴 PNG, 시간·색 패턴 CSV, 표준별 세트 정의 15종(약 23 MB) |
| 원 논문에서의 용도 | 검사 도구가 표준(ITU·Ofcom·ISO·WCAG·trace24)을 지키는지 시험할 **정답이 정해진 영상** 생성. 실패 영상은 문턱을 살짝 넘고, 통과 영상은 한 차원(면적·횟수·휘도·채도·색도)만 문턱에 못 미치게 설계 |
| 라이선스 | BSD-3-Clause |
| 우리가 받은 이유 | 위험 판정기(IRIS, 앞으로 만들 R)의 **단위 시험 세트**. 환자 자료가 아니라 "표준 문서 준수"의 정답이다 |
| 지금까지 한 일 | 6개 세트 110개를 생성해 IRIS로 판정 → 기대 실패 36개 중 3개만 검출, IRIS가 화면 평균으로 판정함을 발견([IRIS_LABELING_PLAN.md](IRIS_LABELING_PLAN.md) 5.2) |
| 위치 | 코드 `_ext\pse-test-media\` · 생성 영상 `_ext\pse-test-media\video_creation\<세트>\videos\*.avi` (FFV1, 30 fps, FHD) · IRIS 결과 `_work\iris_checks\ptm_iris_results.json` |
| 받은 날·방법 | 2026-11-07, `git clone --depth 1`. 영상은 생성기 스크립트를 venv 파이썬으로 직접 실행(원 `make_videos.py`는 Windows에서 venv 밖 파이썬을 호출해 실패) |
| 보관 | 계속 보관(BSD-3, 재생성 가능) |
| 주의 | 생성 영상은 깜빡임 시험 자극. 재생 금지 |

### X3. EA IRIS (검사 도구)와 빌드 의존성

| 항목 | 내용 |
|---|---|
| 출처 | Electronic Arts, https://github.com/electronicarts/IRIS (커밋 `d96978a`, 2025-01-14) · 빌드 의존성 vcpkg(https://github.com/microsoft/vcpkg) |
| 라이선스 | IRIS BSD-3-Clause(저작권 고지 유지, EA 명의 홍보 금지) |
| 우리가 받은 이유 | 합성 데이터의 독립 검사기(라벨러). 데이터가 아니라 도구 |
| 위치 | `_ext\IRIS\` (실행 파일 `bin\build\windows-release\example\IrisApp.exe`), `_ext\vcpkg\` |
| 관련 문서 | [IRIS_LABELING_PLAN.md](IRIS_LABELING_PLAN.md) 5.1–5.3 |

## 2. 받은 문헌 원문 (PDF, 데이터 아님)

위치 `_work\refs\` 및 `_work\refs\papers\`(220 MB). 문헌 정리는 [PAPER_NOTES.md](PAPER_NOTES.md), [PSE_EVIDENCE.md](PSE_EVIDENCE.md).

| 파일 | 문헌 | 받은 곳 |
|---|---|---|
| `refs\barbu2020.pdf` | Barbu, Banda & Katz 2020, *Pattern Recognition Letters* | MIT CSAIL 공개 PDF |
| `refs\banda_thesis_2018.pdf` | Banda 2018, MIT M.Eng. 학위논문 | DSpace@MIT 1721.1/121622 (자동 다운로드가 막혀 사용자가 브라우저로 받음) |
| `refs\osf_south2021\south2021_preprint.pdf` | South 외 2021 사전 공개본 | OSF 4kgu6 |
| `papers\itu_bt1702\bt0-200502.pdf`, `bt1-201807.pdf`, `bt2-201910.pdf`, `v3.pdf` | ITU-R BT.1702 -0 (2005), -1 (2018), -2 (2019), -3 (2023) | itu.int |
| `papers\other\herrick1978_aston_thesis.pdf` | C. E. Herrick 1978, University of Aston in Birmingham 박사 학위논문(광과민성 자극 특성. 깜빡임 면적 관련 1960–70년대 문헌 검토) | 조사 중 공개 PDF로 받음(정확한 URL 미기록) |
| `papers\wilkins_archive\1979-15.pdf` | Wilkins, Darby & Binnie 1979, *Brain* 102:1–25 | visualstress.info (Wilkins 자료실) |
| `…\1980-20.pdf` | Wilkins, Binnie & Darby 1980, *Prog Neurobiol* 15:85–117 | 같음 |
| `…\1981-25.pdf` | Binnie 외 1981, *Advances in Epileptology* | 같음 |
| `…\1985-40.pdf` | Binnie, Findlay & Wilkins 1985, *EEG Clin Neurophysiol* 61:1–6 | 같음 |
| `…\1985-47.pdf` | Wilkins & Lindsay 1985, *Recent Advances in Epilepsy* 2 | 같음 |
| `…\1977-8.pdf` | Stefánsson 외 1977, *BMJ* 2:88–90 | 같음 |
| `…\2001-150.pdf` | Binnie 외 2002, *SMPTE J* 111:323–329 (스캔) | 같음 |
| `…\2005-167.pdf` | EFA 합의문(Harding 외 2005) 저자본 | 같음 |
| `…\2005-168.pdf` | Fisher 외 2005 리뷰 저자본 | 같음 |
| `…\2005-169.pdf` | Wilkins, Emmett & Harding 2005 저자 원고 | 같음 |
| `…\book1.pdf` | Wilkins 1995, *Visual Stress* (스캔) | 같음 |
| `…\wilk1999.pdf` | Wilkins 2010, *Atlas of Epilepsies* 장 | essex.ac.uk |
| `…\wilkins2010.pdf` | Wilkins, Veitch & Lehman 2010, IEEE ECCE | northeastern.edu |
| `…\wind2.pdf` | Smedley, Webb & Wilkins 2010, *Epilepsia* 51:1146–1151 | puc.sd.gov 공개 자료 |
| `…\2011-200.pdf`, `2021-264.pdf`, `2021-265.pdf` | Wilkins 그룹 2011(편두통 fMRI), 2021(편두통 광공포증 리뷰), 2021(패턴 민감 환자의 자기 유발) | visualstress.info |
| `…\1977-10.pdf`, `1979-14.pdf`, `1984-39.pdf`, `1986-52.pdf`, `1987-62.pdf`, `1989-79.pdf`, `1990-87.pdf`, `1995-121a.pdf`, `2010-196.pdf`, `smpte.pdf` | Wilkins 자료실 번호 파일. 스캔본이라 제목을 자동으로 확인하지 못함(1979-14는 Wilkins 외 1979 TV 뇌전증 논문으로 추정) | visualstress.info |

## 3. 우리가 만든 데이터 (재생성 가능)

| ID | 내용 | 만든 명령 | 위치 | 크기 | 용도 |
|---|---|---|---|---|---|
| G1 | 데모 원본 영상 1개(점멸 없음, 3초) | `scripts/generate_demo.py` | `_work\sample\lumi_demo.avi` | 60 KB | 기능 시험 |
| G2 | 합성 원본 30개(점멸 없는 질감·도형·카메라 이동, 8초, 256×192) | `scripts/generate_sources.py --count 30 --seed 1000` | `_work\sources\synth_v1\` | 151 MB | 원본 단위 분할 시험(실제 영상 대용) |
| G3 | Oracle 데이터 1,200클립(16프레임, 128×96) | `build_dataset.py`(G2 30개, `--samples 40 --seed 42`) | `_work\datasets\oracle_s30\` | 778 MB | G 기준선 실험 E1 |
| G4 | IRIS 라벨용 6초 클립 40개 + 무손실 영상 | `build_dataset.py --iris-clips`(G1) | `_work\datasets\iris_v1\` | 11 MB | IRIS 파이프라인 시험 |
| G5 | 동작 확인용 소량 데이터 | `build_dataset.py --samples 8` | `_work\datasets\smoke\` | 0.2 MB | 학습 스크립트 확인 |
| G6 | IRIS 확인 시험 자극·탐색 자극 | `scripts/iris_selftest.py` 등 | `_work\iris_checks\` | 8 MB | IRIS 동작 확인(5.1–5.2) |
| G7 | South 2021 클립을 6초 이상으로 반복한 MKV 188개(깜빡임 94 + 안전 94) | 분석 스크립트(임시) | `_work\datasets\south2021\looped\` | 4.4 GB | 반복 재생 효과 확인(5.3). **언제든 삭제 가능** |
| G8 | South 2021 IRIS 판정 결과 | 분석 스크립트(임시) | `_work\datasets\south2021\iris_original.json` | 0.2 MB | 5.3 |
| G9 | G 체크포인트·학습 로그(λ = 20/2/0) | `scripts/train_gan.py` | `_work\checkpoints\` | 3.5 MB | E1 |

## 4. 삭제 계획

| 대상 | 시점 | 비고 |
|---|---|---|
| `_work\datasets\south2021\` (X1 미디어, G7, G8) | 연구 종료 시 | 사용자 결정. 라이선스 미표기 |
| `_work\datasets\south2021\looped\` (G7) | 공간이 필요할 때 언제든 | 재생성 가능 |
| `OneDrive\Desktop\GAN` (옛 작업 폴더) | Claude Code를 새 위치에서 연 뒤 | 내용은 모두 `C:\dev\lumi-vision`으로 옮김 |
