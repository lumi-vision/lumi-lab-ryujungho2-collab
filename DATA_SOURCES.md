# 데이터 출처 정리

> 기준일: 2026-11-08
> 이 저장소에는 **데이터 파일을 올리지 않습니다.** 이 문서에는 출처, 라이선스, 받는 방법만 적습니다.
> 초기화 전 작업 전체(코드, 문서, 실험 기록)는 브랜치 `archive/2026-11-08-before-reset`와 태그 `before-reset-2026-11-08`에 그대로 남아 있습니다.

## 한눈에 보기

| # | 이름 | 무엇 | 라이선스 | 재배포 | 위험 자극 |
|---|---|---|---|---|---|
| 1 | Blender 오픈 무비 4편 | 깜빡임 없는 일반 영상(애니메이션·실사) | CC BY | 가능(출처 표기) | 없음 |
| 2 | South 외 2021 GIF·영상 2,000개 | 소셜미디어 실제 GIF·영상 + 위험 라벨 | 표기 없음 | **불가** | **있음(163개)** |
| 3 | pse-test-media | 표준 문턱값 시험 영상 생성기 | BSD-3-Clause | 가능 | 생성 영상에 있음 |
| 4 | EA IRIS | 깜빡임 검사 도구(데이터 아님) | BSD-3-Clause | 가능 | 없음 |
| 5 | 논문·표준 문서 PDF | 참고 문헌 | 각 출판사 | **불가** | 없음 |

> **위험 자극이 있는 영상은 재생하거나 미리보기하지 마세요.** 광과민성 발작을 일으킬 수 있습니다. 검사 도구에만 넣습니다.

---

## 1. Blender 오픈 무비 4편

깜빡임 없는 "정상 영상" 원본입니다. 받은 날짜는 2026-11-08입니다.

| 영화 | 파일 | 받는 주소 | 크기(바이트) | 라이선스 |
|---|---|---|---|---|
| Big Buck Bunny (2008) | `BigBuckBunny_640x360.m4v` | https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v.zip (zip 안에 영상 1개) | 121,283,919 | CC BY 3.0 |
| Elephants Dream (2006) | `elephantsdream-480-h264-st-aac.mov` | https://download.blender.org/ED/elephantsdream-480-h264-st-aac.mov | 103,636,531 | CC BY 2.5 |
| Sintel (2010) | `Sintel.2010.1080p.mkv` | https://download.blender.org/durian/movies/Sintel.2010.1080p.mkv | 1,180,090,590 | CC BY 3.0 |
| Tears of Steel (2012) | `tears_of_steel_720p.mov` | https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov | 372,178,639 | CC BY 3.0 |

- 출처 표기: "© Blender Foundation | www.blender.org"
- 검증: 공식 서버가 알려준 크기와 받은 파일 크기를 비교했습니다. Elephants Dream, Sintel, Tears of Steel은 바이트 단위까지 일치했습니다. Big Buck Bunny는 2026-11-08에 서버가 `.m4v` 대신 `.zip`으로만 제공하도록 바뀌었습니다. zip을 다시 받아 풀어 보니 안의 영상 SHA256이 아래 값과 같았습니다.

SHA256:

```
738e2f999860553d056dd79c952f58f63cbb73892a57c72342ce9e5330d9d2d7  BigBuckBunny_640x360.m4v
a7dbacae5fa2bc345dbd7b10588575f3e2bba36012e9a93df5346467053b0634  elephantsdream-480-h264-st-aac.mov
97f1dbc66231df42ad49bd8c29aa174b8f48933058e47e7157d4ba63d93a8efa  Sintel.2010.1080p.mkv
efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f  tears_of_steel_720p.mov
```

다시 받는 법:

```bash
curl -LRO https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v.zip && unzip BigBuckBunny_640x360.m4v.zip
curl -LRO https://download.blender.org/ED/elephantsdream-480-h264-st-aac.mov
curl -LRO https://download.blender.org/durian/movies/Sintel.2010.1080p.mkv
curl -LRO https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov
```

## 2. South, Saffo & Borkin (2021) 소셜미디어 GIF·영상

| 항목 | 내용 |
|---|---|
| 논문 | L. South, D. Saffo, M. A. Borkin, "Detecting and Defending Against Seizure-Inducing GIFs in Social Media", CHI 2021, doi:10.1145/3411764.3445510 |
| 받는 곳 | OSF https://osf.io/5a3dy/ (사전 공개 논문 https://osf.io/4kgu6/) |
| 내용 | Twitter·Tumblr·Tenor·GIPHY에서 500개씩 모은 GIF 1,500개 + MP4 500개(약 3.2 GB), 라벨 CSV `social-media-study.csv` |
| 라벨 | 위험 163 / 안전 1,837. 사람이 붙인 라벨이 아니라 저자가 만든 검출기의 판정입니다. |
| 라이선스 | **표기 없음.** 저작권은 원 게시자에게 있습니다. 내부 연구용으로만 쓰고 재배포하지 않습니다. |
| 사용 규칙 | 위험 163개는 **시험 전용**입니다. 어떤 모델 학습에도 쓰지 않습니다. |
| 받은 날 | 2026-11-07, OSF API로 받음(실패 0건) |

이 자료로 만든 파생 데이터 `real_bg_v1`(안전 영상만 잘라 만든 학습용 배경 클립, 약 2.1 GB)도 같은 이유로 **재배포하지 않습니다.**

## 3. pse-test-media (TRACE RERC)

| 항목 | 내용 |
|---|---|
| 논문 | J. B. Jordan, "Evaluating Conformance of Video Safety Tools for Photosensitive Epilepsy", UAHCI 2025, LNCS 15780:85–98, doi:10.1007/978-3-031-93848-1_7 (무료 저자본 PMC12249941) |
| 받는 곳 | https://github.com/traceRERC/pse-test-media (커밋 `ef78542`) |
| 내용 | ITU·Ofcom·ISO·WCAG 기준에 맞춰 "정답이 정해진" 깜빡임 시험 영상을 만드는 생성기 |
| 라이선스 | BSD-3-Clause |
| 주의 | 생성된 영상은 깜빡임 자극입니다. 재생하지 마세요. |

## 4. EA IRIS (검사 도구)

| 항목 | 내용 |
|---|---|
| 받는 곳 | https://github.com/electronicarts/IRIS (커밋 `d96978a`), 빌드에 vcpkg(https://github.com/microsoft/vcpkg) 필요 |
| 용도 | 영상의 깜빡임 위험을 판정하는 도구. 데이터가 아니라 도구입니다. |
| 라이선스 | BSD-3-Clause(저작권 고지 유지, EA 이름으로 홍보 금지) |

## 5. 참고 문헌 PDF

PDF는 저작권 때문에 올리지 않습니다. 목록과 받은 곳은 아카이브 브랜치의 [`docs/DATASETS.md` 2절](https://github.com/lumi-vision/lumi-lab-ryujungho2-collab/blob/archive/2026-11-08-before-reset/docs/DATASETS.md)에 있습니다. 주요 문헌은 다음과 같습니다.

- Barbu, Banda & Katz 2020, *Pattern Recognition Letters* (깜빡임 생성 GAN)
- Banda 2018, MIT M.Eng. 학위논문 (DSpace@MIT 1721.1/121622)
- ITU-R BT.1702 -0 ~ -3 (itu.int)
- Wilkins 그룹 광과민성 논문들 (visualstress.info)

## 6. 우리가 만든 데이터

모두 아카이브 브랜치의 스크립트로 다시 만들 수 있어서 따로 보관하지 않습니다.

| 이름 | 내용 | 비고 |
|---|---|---|
| `synth_v1` | 깜빡임 없는 합성 영상 30개 | 재생성 가능 |
| `oracle_s30` | 합성 영상으로 만든 학습 클립 1,200개 | 재생성 가능 |
| `iris_v1`, `smoke` | 파이프라인 시험용 소량 데이터 | 재생성 가능 |
| `checkpoints/g_*.pt` | 이전 G 모델 가중치(약 21 MB) | 새 설계에서는 쓰지 않음 |
