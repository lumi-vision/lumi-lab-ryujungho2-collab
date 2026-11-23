# '실제 위험'과 외부 기준 — PEAT·Kaya 확인 (2026-11-23)

목적: 판정기 v2 를 평가할 때 **우리 판정기와 독립된 기준**을 무엇으로 삼을지. (순환 문제: 우리 판정기로 고른 '위험 영상'으로 우리 판정기를 평가할 수 없다.)

## 1. PEAT (Trace Center, UMD) — 원문 확인

| 항목 | 내용 |
|---|---|
| 배포 | 무료. v1.6, `https://trace.umd.edu/wp-content/uploads/2021/01/peat_2017-02-15.zip` |
| 환경 | Windows GUI (공식 표기 Vista/7/XP, Windows 10 은 일부). 명령줄·일괄 처리 언급 없음 |
| 입력 | AVI, 1024×768 전체 화면 캡처 기준, 25 또는 30 fps, 2 GB 미만 |
| 판정 | WCAG 2.0 기준 일반 섬광·적색 섬광 + 연장 섬광 경고. **공간 패턴 판정 없음** |
| 엔진 | PSEe 엔진(Cambridge Research Systems, 비공개), Harding 박사와 공동 개발 |
| 출력 | HTML 보고서·PEAT 분석 파일. **프레임별 내보내기 문서화 안 됨** |
| **사용 제한** | "Use of PEAT to assess material commercially produced for television broadcast, film, home entertainment, or gaming industries is prohibited." → 방송·영화에서 온 영상에는 쓸 수 없음 |
| 향후 | 완전 오픈소스판 준비 중, 일정 없음 |

**결론**: 자동화가 어렵고(GUI), 패턴이 없고, 방송·영화 영상에 쓸 수 없다 → **주 기준으로는 부적합**. 웹·사용자 영상과 합성 영상의 소량 교차 확인용으로만.

## 2. Kaya 2025 — 기존 조사(`docs/research/kaya_dataset.md`, 11-10)와 압축 해제본 확인

- 논문: Kaya, Kilic, Genc, Kural (2025), *Signal, Image and Video Processing* 19(12):967 (본문 유료, 미확인).
- **수집 기준**(같은 연구실 석사논문 Akyıldız 2025, ★원문 — 같은 데이터인지는 미확인): 온라인 방송·인터넷 아카이브·사용자 영상에서 직접 모음, **"only publicly available videos containing appropriate warnings were included"**, 게임 제외, 합성 섬광은 '인위적'이라 제외.
- **라벨**: "PEAT was used as a practical proxy for labeling"(석사논문). 배포본 `out/<n>/correction_info.txt` 는 저자들 자체 탐지기(W3C/Ofcom 규칙)의 결과 — 예: 1 번 원본 위험 98.65 %, 구간 `general 7–518`.
- 보충자료: 응답자 42 명이 영상 18 개를 1–5 점으로 평가한 설문.

**결론**: Kaya 의 '위험' 선정 기준은 **"외부(제작·배포자)가 광과민성 경고를 붙인 공개 영상"** — 우리 판정기와 독립. 라벨은 PEAT/자체 규칙 판정이지 임상 정답이 아니다.

## 3. 제안 — 평가 기준을 셋으로

| 질문 | 기준 | 자료 |
|---|---|---|
| 규격을 정확히 구현했나 | 정답이 규격에서 나오는 시험 영상 | 깜빡임: pse-test-media(TRACE RERC, Jordan 2025) · **패턴: pse-test-media 에 정지 줄무늬 시험이 없음(면적 패턴은 깜빡임 면적용)** → 표준 조건으로 직접 만든 경계 시험 세트 + hazard7 hazard/near 쌍 |
| 독립 구현과 일치하나 | 자동화 가능한 오픈소스 판정 도구 | **EA IRIS**(BSD-3, 기존에 사용 — `DATA_SOURCES.md` §4), PEAT 는 웹·합성 소량 확인만 |
| 실제 위험·일상 영상에서 | 위험 = **외부 경고가 붙은 공개 영상**(Kaya 와 같은 선정 기준) · 일상 = South 안전 | Kaya 22 (+ 같은 기준으로 확장 시 저작권상 내부 분석만) · South 1,837 |

패턴은 PEAT·IRIS 모두 다루지 않고 pse-test-media 에도 줄무늬 시험이 없다(11-23 정정) → 표준 조건(ITC/Ofcom·BT.1702-3 Att.1)으로 직접 만든 경계 시험 세트와 사람 확인이 기준이다.
