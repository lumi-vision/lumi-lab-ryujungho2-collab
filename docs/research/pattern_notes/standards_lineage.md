# 방송·접근성 표준의 광과민성 "규칙 패턴(regular/spatial pattern)" 정의와 계보

> 표기 규칙: **[원문]** = 표준 원문을 직접 확인, **[2차]** = 다른 문헌을 통해 확인, **[미확인]** = 확인하지 못함.
> 로컬 원문 사본: `scratchpad/dl/` (bt1702-0/1/2/3.txt, gnflash.txt, ofcom2.txt, wcag20/21/22.html, und22.html, jv24.txt, pseg.md), `scratchpad/jp/jp.txt` (NHK/민방련 지침), `scratchpad/iso/iso.txt` (ISO/DIS 9241-391.2 미리보기).

---

## Q0. 요약 비교표 (표준별 패턴 정의·수치)

| 표준 / 판 | 패턴 정의 원문 (요지) | 줄무늬 쌍 수 | 대비/휘도 | 면적: 정지 | 면적: 변화(진동·점멸·반전·방향 전환) | 지속시간 | 공간주파수 / 시거리 | 운동 면제 | 규범성 | 계보(복사 원천) |
|---|---|---|---|---|---|---|---|---|---|---|
| ITC Guidance Note 1994 (초판) | [미확인] 원문 확보 실패 | [미확인] | [미확인] | [미확인] | [미확인] | – | – | [미확인] | 면허 조건 지침 | Pot Noodle 광고 사건(1993) 이후 제정 [2차] |
| ITC GN 1999 개정 | [미확인] 원문 확보 실패 | [미확인] | [미확인] | [미확인] | [미확인] | – | – | [미확인] | 동일 | 1994판 |
| **ITC GN 2001 (2002 편집 수정)** [원문] | "A potentially harmful regular pattern contains clearly discernible stripes when there are more than five light-dark pairs of stripes in any orientation…" | **>5쌍** | 어두운 막대 **<160 cd/m²** 이고 밝은 막대와 **≥20 cd/m²** 차이 (감마 2.2, 피크화이트 200 cd/m², BT.500 가정) | **>40%** 화면 | **>25%** 화면 | 없음 | 없음 (시거리·cpd 규정 없음) | "obviously flow smoothly across, into, or out of the screen in one direction" → 면제 | 지침(Guidance note) | ITC 자체 + Harding/Wilkins 의견 (참고문헌: Harding & Jeavons 1994, Harding 1998, Wilkins 1995) |
| **Ofcom Guidance Notes Sec.2, Annex 1** (2005 재발행, 2017 Issue 12) [원문] | ITC 2001 문구 **글자 그대로** 승계 | >5쌍 | ITC와 동일 (<160 & ≥20) | >40% | >25% | 없음 | 없음 | 동일 문구 | Code Rule 2.12의 해석 지침 | ITC GN 2001 |
| ITU-R BT.1702-0 (2005) [원문] | 패턴 기준 **없음**. 서문에 "certain types of regular pattern can cause problems"만 언급 | – | – | – | – | – | – | Annex/Appendix의 측정 프레임워크 그림에 "Motion compensation reduces spurious detections due to panning and object motions" | 권고 | ITC GN 2001(플래시 부분 거의 그대로) |
| BT.1702-1 (07/2018) [원문] | 패턴 기준 없음 (동일 서문) | – | – | – | – | – | – | – | 권고 | -0 |
| BT.1702-2 (10/2019) [원문] | 패턴 기준 없음. HDR 1/17 Michelson 규칙 **처음 도입** (플래시에만) | – | – | – | – | – | – | – | 권고 | -1 |
| **BT.1702-3 (11/2023)** [원문] | Guideline 2: "Potentially harmful regular patterns occur when an image contains light and dark pairs of clearly discernible stripes in any orientation…" | 본문(Guideline 2)에는 수치 없음; **Attachment 1(informative)**: ">5쌍" | Attachment 1: Guideline 1의 플래시 대비와 **동일** → <160 cd/m²는 20 cd/m², ≥160은 **>1/17 Michelson(HDR만)** | Attachment 1: >40% | Attachment 1: >25% | 없음 | 없음. Annex 5: 시청 거리 "at least two metres" 권장(시청자 조언) | Attachment 1(informative)에만: "If the patterns obviously flow smoothly across, into, or out of the screen in one direction they are exempt from restriction." | Guideline 2 = Annex 1 본문(권고), Attachment 1 = **informative, "used by some administrations"** | ITC/Ofcom 문구 (거의 복사) |
| **NHK·민방련 「アニメーション等の映像手法について」** 1998-04-08 작성, 2006-04-01 일부개정 [원문] | "３．規則的なパターン模様（縞模様、渦巻き模様、同心円模様など）が、画面の大部分を占めることも避ける。" | 없음 | 없음 | "画面の大部分" (정량 없음) | 구분 없음 | 없음 | 없음. "明るい部屋で受像機から離れて見る" 조언 | 없음 | 업계 자율 공통 규칙 ("遵守しなければならない") | 1998 독자 제정(포켓몬 사건), 2006년 BT.1702-0 참고 개정 |
| NHK·민방련 2020 개정 | [미확인] (JV24는 2020 갱신·HDR 수치 추가를 보고 [2차]; 패턴 문구 변경 여부 미확인) | | | | | | | | | |
| **ISO 9241-391:2016** (2021 재확인) | 5.3.1 "Potentially harmful regular patterns" 조항 존재 [원문: 목차]. 조항 본문 [미확인] | [미확인] (JV24는 ISO가 Ofcom과 같은 면제 문구를 가진다고 기술 [2차]) | [미확인] | [미확인] | [미확인] | [미확인] | [미확인] | "across, into, or out of the screen in one direction" 면제 있음 [2차: JV24] | 국제표준(요구사항) | ISO/IWA 3:2005 → TC159/SC4; NOTE로 BT.1702, WCAG 2.0 참조 [원문 DIS] |
| **WCAG 2.0 (2008) / 2.1 (2018) / 2.2 (2023)** SC 2.3.1 [원문] | 패턴 SC **없음**. 플래시 임계의 예외로만: "Flashing that is a fine, balanced, pattern such as white noise or an alternating checkerboard pattern with "squares" smaller than 0.1 degree (of visual field at typical viewing distance) on a side does not violate the thresholds." | – | – | – | – | – | **0.1° 정사각(=6 arcmin)**; 10° 시야 = 1024×768에서 341×256 px, "15-17 inch screen at 22-26 inches" | – | 규범(SC) | UK TV 사양을 컴퓨터용으로 각색 (Understanding 2.3.1) |
| WCAG 2.0 초안 (2004-02-17 제안) [2차: 메일링리스트] | "Clearly discernible stripes consisting of more than five light-dark pairs in any orientation are not permitted when the stripes are stationary" | >5쌍 | <160 & ≥20 cd/m² | 355×268 px 사각형(1024×768)의 >40% | 같은 사각형의 >25% | – | 355×268 px | [미확인] | 제안(Level 3) – 최종 2.0에서 삭제 | ITC GN 2001 + "Wisconsin Computer Equivalence Algorithm" |
| **Xbox XAG 118** (ms.date 2022-05-09) [원문] | "Alternating bands that have high contrast define a spatial pattern." | **쌍 수 없음** | "difference in contrast is greater than 10 percent" | "approximately 20 percent or more" (정지/변화 구분 없음) | 동일 | – | – | – | 플랫폼 가이드라인 | 독자 단순화; Harding FPA 링크 |
| Harding FPA (벤더 해설 페이지) [원문: 벤더] | "pairs of alternating light-dark bars of at least 20cd/m2 contrast"; "6 or more stationary pairs which persist for more than 0.5 seconds" & ">40%" | ≥6쌍 | ≥20 cd/m² | >40% | 페이지에 명시 없음 | **>0.5 s 정지 지속** | – | 카메라 팬/줌 등 운동 → Ofcom 기준상 면제 | 상용 도구 | Ofcom |
| trace24 (Jordan & Vanderheiden 2024, pseGuidelines) [원문] | 제안 provision은 **플래시 전용**; 패턴 조항 없음 | – | – | – | – | – | 시거리 표(Table 2) 제시 | "no specific, actionable guidance available on motion compensation thresholds" | 제안(informative) | ITU/Ofcom/ISO/NHK/WCAG 통합 |

---

## Q1. ITU-R BT.1702 (-0 2005, -1 2018, -2 2019, -3 2023): 패턴 지침은 언제·어떻게 등장했나

### Takeaway
BT.1702-0/-1/-2 에는 패턴 기준이 **전혀 없고**(서문에 "certain types of regular pattern" 언급뿐), 패턴 지침은 **BT.1702-3(11/2023)** 에서 처음 들어왔다. 정성적 정의(Guideline 2)는 Annex 1 본문에, 정량 수치(>5쌍, 40%/25%, 대비)와 "flow smoothly … in one direction" 면제는 **informative Attachment 1**("used by some administrations")에 분리되어, 사실상 ITC/Ofcom 문구를 옮긴 것이다. HDR 1/17 Michelson 규칙은 **-2(2019)** 에서 플래시용으로 도입됐고, -3의 Attachment 1이 패턴 대비를 "Guideline 1과 동일"로 규정함으로써 패턴에도 간접 적용된다.

### Cited Findings
- [원문] BT.1702-0 표제 연도 "(2005)", Question ITU-R 47/6. Scope: "prone to seizures triggered by flashing lights, including certain types of flashing television images" — 패턴 언급 없음. Appendix 1 제목 "Technical information for production organizations on flashing images in television", 첫 문장 "Flickering or intermittent images and certain types of regular pattern can cause problems…", 이후 플래시 기준만 존재 — [ITU-R BT.1702-0](https://www.itu.int/rec/R-REC-BT.1702-0-200502-S/en) (로컬 bt1702-0.txt)
- [원문] -0의 측정 프레임워크 그림(Fig. 2) 주석: "If target of processing is fields, fine horizontal stripes can give flicker indication"; "Appropriate block averaging reduces spurious detections due to noise and fine patterns"; "Motion compensation reduces spurious detections due to panning and object motions" — 패턴은 *오검출 요인*으로만 다뤄짐 — [BT.1702-0](https://www.itu.int/rec/R-REC-BT.1702-0-200502-S/en)
- [원문] -0의 플래시 정의는 ITC 2001과 거의 동일: "of 20 cd/m2 or more … This applies only when the screen luminance of the darker image is below 160 cd/m2", "more than one quarter of the displayed … screen area", "nine frames or more … 50 Hz … ten frames or more … 60 Hz" — [BT.1702-0](https://www.itu.int/rec/R-REC-BT.1702-0-200502-S/en)
- [원문] BT.1702-1 표지 "(07/2018)", 연혁 "(2005-2018)"; Annex 1 제목 "…on flashing images in television"; 패턴 기준 없음 (grep 결과 "regular pattern"은 서문·Annex 2 문장뿐) — [BT.1702-1](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] BT.1702-2 표지 "(10/2019)", 연혁 "(2005-2018-2019)". HDR 규칙 최초 도입: "When screen luminance of the darker image is not less than 160 cd/m2, a potentially harmful sequence of flashes occurs where there is a difference of more than 1/17 Michelson contrast … (LHIGH - LLOW)/(LHIGH + LLOW) … This is applicable for HDR programmes only." 시간 기준도 "360 ms or more … 50 Hz … 334 ms or more … 60 Hz"로 변경. 패턴 기준은 여전히 없음 — [BT.1702-2](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] -2 NOTE 5: "There is no discontinuities at 160 cd/m2 since … 1/17 = ((160 + 20) - (160))/((160 + 20) + 160)." — [BT.1702-2](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] BT.1702-3 표지 "(11/2023)", 연혁 "(2005-2018-2019-2023)". Scope에 "or rapidly changing images or certain types of regular patterns" 및 각주 "Sometimes referred to as spatial patterning." 추가; Keywords에 "regular patterns"; considering a)에 "and regular patterning" 추가; Annex 1 제목이 "Guidelines for production organizations on flashing **or patterned** images in television"으로 변경 — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] BT.1702-3 **Guideline 2** 전문: "Potentially harmful regular patterns occur when an image contains light and dark pairs of clearly discernible stripes in any orientation. The stripes may be parallel or radial, curved or straight, and may be formed by rows of repetitive elements such as polka dots. If the stripes change direction, oscillate, flash or reverse in contrast they are more likely to be harmful than if they are stationary. Attachment 1 to this Annex provides informative guidance for measurement to identify potentially harmful patterns." — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] BT.1702-3 **Attachment 1 to Annex 1 (informative)** "Guidance on measurement criteria to identify potentially harmful patterns": "The following additional guidance on the identification of potentially harmful regular patterns is used by some administrations. Potentially harmful regular patterning may occur if an image contains more than five light and dark pairs of clearly discernible stripes in any orientation when either of the following two conditions apply: - the stripes are stationary and the pattern occupies more than 40% of the displayed screen area; or - the stripes change direction, oscillate, flash, or reverse in contrast and the pattern occupies more than 25% of the displayed screen area. For a pattern to be considered potentially harmful, the difference between the luminance of the darker and lighter bars of the pattern will be the same as the difference between the darker and brighter images of a potentially harmful flash described in Guideline 1 above. If the patterns obviously flow smoothly across, into, or out of the screen in one direction they are exempt from restriction." — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] -3 Guideline 1 (패턴 대비가 참조하는 플래시 대비): darker <160 cd/m² → "difference of 20 cd/m2 or more … applicable for both SDR and HDR"; darker ≥160 → "more than 1/17 Michelson contrast … applicable for HDR programmes only". NOTE 3: SDR 피크화이트 200 cd/m², HLG 1 000 cd/m²; Annex 2: PQ 10 000 cd/m². 면적 표현이 "one quarter"에서 "25%"로 변경 — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] -3 Annex 2 Table 1 NOTE 1/2: "If the brighter image of a flash **or pattern** is above this level…", "If the darker image of a flash **or pattern** is above this level, then it is potentially harmful if the screen luminance whose Michelson luminance contrast is 1/17 or more." — 즉 1/17 규칙이 패턴에도 명시적으로 적용됨 — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] -3 참고문헌에 패턴 관련 [9] El Shakankiry & Abdel Kader "Pattern sensitivity: a missed part of the diagnosis" (PMC3404594), [10] Wilkins "Pattern-Sensitive Epilepsy" (essex.ac.uk 1999-141.pdf), [11] Pelli & Bex 2013(Michelson 정의) 추가 — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] -3 Annex 5: "consideration should be given on providing advice for watching television in a well-lit room from a distance of at least two metres" — 시거리 가정은 시청자 조언일 뿐 측정 기준 아님 — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] -3 Annex 3(측정 프레임워크): "it needs further study to develop a unified specification … Schematics and detailed definitions for each block will be required." — [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [2차] Jordan & Vanderheiden 2024 Table 1은 ITU-R 갱신일을 "2005, 2018, 2019, and 2023"으로 표기 — [JV24, PMC11872230](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)

### Inferences
- 규범성: ITU-R Recommendation 자체가 비구속 권고이며, 그 안에서도 **Guideline 2(정성 정의)는 Annex 1 본문 = 권고 지침**, **수치(>5쌍·40%/25%·대비)와 smooth-flow 면제는 "informative" 부속서**로 격하·분리되어 있다. 즉 ITU는 패턴의 정량 기준을 "일부 주관청이 사용하는" 참고로만 채택.
- -3 Attachment 1 문장은 Ofcom Annex 1 §5/5.1과 단어 수준으로 일치("change direction, oscillate, flash, or reverse in contrast", "obviously flow smoothly across, into, or out of the screen in one direction") → 직접 복사로 판단. 차이점: ① ">5쌍"이 정의문에서 조건문으로 이동, ② 대비 조건이 "<160 & ≥20"에서 "Guideline 1과 동일"(→ HDR ≥160 cd/m²에 1/17 Michelson 확장)로 일반화, ③ 면제 문장이 Guideline 2 본문에서 informative 부속서로 이동.
- 파일명/URL에 "202302"가 쓰였으나 PDF 표지는 "(11/2023)"이다. 승인/발행 월 불일치 가능성 — 확인 필요(아래 Gaps).

### Gaps
- BT.1702-3 승인일: PDF 표지는 11/2023, ITU URL 관례(R-REC-BT.1702-3-2023xx)상 정확한 월은 [미확인] (로컬 202302 파일은 245바이트 실패 다운로드).
- -3 개정 경위(어느 주관청 기고로 패턴이 추가됐는지, ITU-R WP6C 문서) [미확인].

---

## Q2. UK ITC Guidance Note (1994 → 1999 → 2001/2002) → Ofcom (2005 재발행, 2017 Annex 1)

### Takeaway
현존 확인 가능한 최古 원문은 ITC GN 2001(2002 편집 수정)이며, 패턴 정의(">5 light-dark pairs", "parallel or radial, curved or straight", "polka dots", 40%/25%, <160 & ≥20 cd/m², smooth-flow 면제)는 Ofcom이 2005년 그대로 재발행해 2017 Issue 12 Annex 1까지 **문구 변경 없이** 유지됐다. 1994 초판·1999 개정의 패턴 문구는 원문을 확보하지 못했다.

### Cited Findings
- [원문] ITC GN 표지: "Revised and re-issued July 2001 / Editorial amendment June 2002"; 말미: "First issued as an ITC Guidance Note November 1994, first revised September 1999, revised and re-issued July 2001." — [ITC gn_flash.pdf (Ofcom 보관 LEGACY DOCUMENT)](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [원문] ITC §5: "A potentially harmful regular pattern contains clearly discernible stripes when there are more than five light-dark pairs of stripes in any orientation. The stripes may be parallel or radial, curved or straight, and may be formed by rows of repetitive elements such as polka dots. If the stripes change direction, oscillate, flash or reverse in contrast they are more likely to be harmful than if they are stationary. If the patterns obviously flow smoothly across, into, or out of the screen in one direction they are exempt from restriction." — [ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [원문] ITC §5.1: "Potentially harmful patterns are not permitted when either of the following two conditions apply: i. the stripes are stationary and the pattern occupies more than 40% of the displayed screen area; or ii. the stripes change direction, oscillate, flash, or reverse in contrast and the pattern occupies more than twenty five per cent of screen area; and in addition to either of the above two conditions applying, when iii. the screen luminance of the darker bars in the pattern is below 160 cd.m-2 and differs from the lighter bars by 20 cd.m-2 or more (see notes 1 and 2)." — [ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [원문] ITC 각주: 감마 2.2 가정; "pictures are assumed to be displayed in accordance with the `home viewing environment' described in Recommendation ITU-R BT.500 in which peak white corresponds to a screen illumination of 200 cd.m-2"; overscan "3.5% ± 1% … (as indicated in EBU Technical recommendation R95-2000)". Appendix I: 234 mV ≈ 20.1 cd/m², 631 mV ≈ 160 cd/m² — 각 노트가 "flash **or pattern**"에 적용 — [ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [원문] ITC 참고문헌: ITC Programme Code Autumn 1998 Section 7.3 (rev. Sept 1999), Section 1.12(iii) (rev. Spring 2001); Harding & Jeavons 1994; Harding 1998 Nature Medicine; Wilkins 1995 Visual Stress — [ITC GN 2001](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [원문] Ofcom Guidance Notes Section 2 "Issue Twelve: 18 July 2017", Rule 2.12 Flashing Images 지침: "Flashing images and/or patterns can cause seizures in people with photosensitive epilepsy. The Ofcom guidelines, based on scientific research, are intended to limit the incidences of seizures and a warning should only be used in place of the guidelines if editorially justified. See attached Annex 1" — [Ofcom Section 2 Guidance Notes](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/programme-guidance/broadcast-code-guidance/section-2-guidance-notes.pdf)
- [원문] Ofcom Annex 1 "Ofcom Guidance Note on Flashing Images and Regular Patterns in Television — Re-issued as Ofcom Notes (25 July 2005) / Revised and re-issued by the ITC in July 2001 / Editorial amendment June 2002". §5, §5.1(i)(ii)(iii)는 ITC와 동일 문구(“polka dots”, "parallel or radial, curved or straight", 40%, "twenty five per cent", <160 & ≥20 cd.m-2, smooth-flow 면제). 단, 말미에 "Re-issued by Ofcom (date tbc)"라는 자체 모순 표기 잔존 — [Ofcom Section 2 Guidance Notes](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/programme-guidance/broadcast-code-guidance/section-2-guidance-notes.pdf)
- [원문] Ofcom판 차이: §1 "The television licensees, together with the ITC, have consulted…" → "These guidelines have been drawn up following consultation with leading medical opinion"; §2 "licensees" → "broadcasters"; 플래시 시간 기준 "9 frames or more"(50 Hz 전제) 유지 — [Ofcom](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/programme-guidance/broadcast-code-guidance/section-2-guidance-notes.pdf); [ITC](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [2차] JV24 Table 1: Ofcom(ITC) 갱신일 "1994, 1999, 2001, 2002, 2005ᵃ, and 2017ᵃ", ᵃ="Reaffirmed or republished with no substantive changes to PSE guidance"; 1993 Golden Wonder Pot Noodle 광고가 UK 지침 제정 계기 — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [2차] Binnie et al. (SMPTE J 2002)의 연구가 "a revision, in mid-2001, of a Guidance Note issued by the ITC"에 기여 — [SMPTE Journal 111(6-7)](https://journal.smpte.org/periodicals/SMPTE%20Journal/111/6-7/10/) (검색 스니펫 기준)

### Inferences
- ITC 2001 패턴 조항의 수치(>5쌍, 40%/25%, 20 cd/m²)가 이후 ITU-R BT.1702-3 Attachment 1, WCAG 2.0 2004 초안, Harding FPA의 직접 원천이다.
- 1994→1999→2001 사이에 패턴 문구가 어떻게 바뀌었는지는 원문 부재로 판단 불가.

### Gaps
- ITC 1994 초판(및 "1994 draft")·1999 개정판 원문 [미확인]: Wayback/ofcom 사이트에서 찾지 못함. 영국 국립문서보관소(UK Government Web Archive) ITC 사이트 아카이브를 추가 탐색할 가치 있음.
- 현행 Ofcom Broadcasting Code Rule 2.12 조문 자체("…take precautions to maintain a low level of risk…")는 로컬 문서에 없어 [미확인].
- 2017년 이후(2019, 2020-21 Code 개정) Annex 1 재발행 여부 [미확인].

---

## Q3. ISO 9241-391:2016 의 패턴 요구사항

### Takeaway
ISO 9241-391:2016(2021 재확인)은 5.3 "Patterns" / 5.3.1 "Potentially harmful regular patterns" 조항을 가진다는 것은 목차로 확인되지만, 조항 본문(쌍 수·면적·대비)은 유료 원문이라 **확인하지 못했다**. JV24는 ISO와 Ofcom이 모두 "across, into, or out of the screen in one direction" 면제를 둔다고 2차로 보고한다.

### Cited Findings
- [원문: 목차] "5.3 Patterns … 5.3.1 Potentially harmful regular patterns"; 5.2 Flashes(5.2.1 harmful flashes, 5.2.2 rapid changes of image sequences, 5.2.3 red flashes, 5.2.4 cumulative risk, 5.2.5 prior warning); 6.2 Test methods — [SIST EN ISO 9241-391:2016 미리보기 (oSIST prEN ISO 9241-391:2014 = ISO/DIS 9241-391.2)](https://cdn.standards.iteh.ai/samples/37651/9af06303a47b4841ae380ed118cff532/SIST-EN-ISO-9241-391-2016.pdf)
- [원문: DIS 미리보기] 서론: "in 2004, the ISO/International Workshop on Image Safety was held, and resulted in the publication of ISO/IWA3:2005 … the study group, (ISO/TC 159/SC 4/SG on Image Safety), continued to evaluate strategies … until 2009." — 위 동일 출처
- [원문: DIS] Scope: "for the protection of the vulnerable individuals in the viewing population who are photosensitive, and who are therefore liable to seizures triggered by flashing lights and regular patterns, including certain repetitive images. NOTE 1 ITU considers the image safety issues in relation to broadcasting. Some of these are described in ITU-R BT.1702 … NOTE 2 … ISO/IEC 40500:2012 (W3C WCAG 2.0)". 규범 참조: ISO 9241-302, ITU-R BT.500-11 — 위 동일 출처
- [원문: DIS] 4절: "Another potentially provocative visual stimulus for some individuals with photosensitivity is regular patterns with a certain numbers of bright stripes per a certain area [6]. The pattern must also fill a large part of the visual field, while the provocation depends on whether the patterns are stationary or oscillating (including flashing) [7][16][17]." 정의 3.3 PPR: "EEG response to flash or pattern"; 3.4 photosensitivity: "…and/or visual patterns" — 위 동일 출처
- [2차] JV24: "The ISO and Ofcom guidelines give specific PSE guidance for patterns and state that potentially hazardous patterns are exempt if they exhibit smooth flowing motion "across, into, or out of the screen in one direction" [49, 69]." — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [2차] JV24 Table 1/3: ISO 플래시 기준 = "≥20 cd/m² when darker state <160 cd/m²", ">25% of displayed screen area", ">3 flashes within 1 s"; 갱신 "2016, 2021ᵃ"(재확인); ISO만 65 Hz 이상을 안전 상한으로 둠; 적색 정의 R/(R+G+B)≥0.8 & Δu'v'>0.2 (WCAG 2.2 Understanding이 [ISO 9241-391] 인용) — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/); [Understanding 2.3.1](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)
- [2차] ISO/CD 9241-391 개정 작업 진행 중(ISO 프로젝트 94287), 2016판 대체 예정 — [ISO 94287](https://www.iso.org/standard/94287.html) (검색 스니펫; 단계·날짜 [미확인])

### Inferences
- 플래시 수치가 ITU/Ofcom과 동일하고 면제 문구가 Ofcom과 동일하다는 2차 정보로 볼 때, 5.3.1도 ITC/Ofcom의 >5쌍·40%/25%·<160&≥20 cd/m² 구조를 복사했을 가능성이 높다 — **추정, 미검증**.

### Gaps
- 5.3.1 본문 수치(쌍 수, 면적, 대비, HDR 처리, 시거리 가정, cpd 유무) 전부 [미확인]. ISO OBP(iso.org/obp) 미리보기는 일반적으로 Scope~용어까지만 공개되어 5.3.1 접근 불가로 판단(미시도분 포함).
- ISO/DIS·CD 개정판에서 패턴 조항 변화 [미확인].

---

## Q4. 일본 NHK·민방련(NAB-J/JBA) 지침 (1998, 2006, 2020)

### Takeaway
포켓몬 사건(1997-12) 이후 1998-04-08 제정, 2006-04-01 BT.1702-0을 참고해 일부개정된 공통 지침의 패턴 조항은 "規則的なパターン模様（縞模様、渦巻き模様、同心円模様など）が、画面の大部分を占めることも避ける" 한 줄이며 **정량 수치가 전혀 없다**(쌍 수·대비·면적% 모두 없음, 정지/운동 구분 없음).

### Cited Findings
- [원문] 표제·날짜: "アニメーション等の映像手法について 1998 年 4 月 8 日 作成 / 2006 年 4 月 1 日 一部改訂 / 日本放送協会 (社)日本民間放送連盟" — [tcnet.ne.jp 별지 PDF](https://www.tcnet.ne.jp/wp-content/uploads/community_ch/broadcasting-plan-standard/standard_Exhibit.pdf)
- [원문] 전문 주의 3항목: "１．映像や光の点滅、特に「鮮やかな赤」の点滅 ２．コントラストの強い画面の反転や急激な場面転換 ３．規則的なパターン模様の使用" — 위 동일
- [원문] 개정 경위: "ＩＴＵ…において、2005 年２月にＩＴＵ－Ｒ勧告ＢＴ．１７０２ … が成立したことから、同勧告を参考にガイドラインを一部改訂することとした。" — 위 동일
- [원문] 지침 본문 3: "３．規則的なパターン模様（縞模様、渦巻き模様、同心円模様など）が、画面の大部分を占めることも避ける。" — 위 동일
- [원문] 플래시 기준(대비용): "点滅が同時に起こる面積が画面の１/４を超え、かつ、輝度変化が 10 パーセント以上"; 조건 충족 시 "１秒間に５回を限度とし、かつ、輝度変化を 20 パーセント以下 … 連続して２秒を超える使用は行わない"; 장면전환 "輝度変化が 20 パーセントを超える…１秒間に３回を超えて使用しない" — 위 동일
- [원문] 시청환경: "明るい部屋で受像機から離れて見るなど"テレビの見方"に関する適切な情報を視聴者に提供することは予防手段として有効" — 위 동일
- [2차] JV24: NHK와 JBA가 "joint guidance, last updated in 2020"; Table 1 갱신 "1998, 2006, and 2020"; HDR용 20–40 cd/m², 1/8–1/4 밝기 변화 등 수치는 2020판에 있다고 보고. JV24는 일본 기준이 아날로그 IRE 단위로 감마 보정 없이 분석된다고 Hodgetts 2014를 인용 — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/) (출처 [55] = https://www.j-ba.or.jp/category/broadcasting/jba103852, 현재 404)
- [2차] NHK·NAB 공동 지침에 기반한 점멸·패턴 검출 장치 연구 존재 — [ITE Technical Report 22.40](https://www.jstage.jst.go.jp/article/itetr/22.40/0/22.40_1/_article/-char/ja) (검색 스니펫)

### Inferences
- 일본 지침의 패턴 조항은 ITC/ITU와 계보가 독립적(1998 자체 제정)이며, 2006 개정 때 BT.1702-0을 참고했으나 BT.1702-0에는 패턴 수치가 없었으므로 패턴 조항이 정량화될 근거가 없었다고 해석된다.
- "縞模様、渦巻き模様、同心円模様"은 ITC의 "parallel or radial, curved or straight"와 개념상 대응(동심원=curved, 渦巻き=radial/curved).

### Gaps
- 2020 개정판 원문 [미확인] (j-ba.or.jp 링크 404). 패턴 조항 문구가 2020년에 바뀌었는지 미확인.
- 1998 원판 패턴 문구(2006 개정 전) [미확인] — 2006판 서문상 일부 개정이므로 동일 가능성 높으나 검증 안 됨.

---

## Q5. WCAG 2.x: 미세 균형 패턴 예외(<0.1°)와 패턴 SC 부재의 이유

### Takeaway
WCAG 2.0(2008)·2.1(2018)·2.2(2023)의 SC 2.3.1 "general flash and red flash thresholds"는 패턴을 **플래시 예외로만** 다룬다(0.1° 미만 미세 균형 패턴). 그러나 2004년 2월 Vanderheiden의 WCAG 2.0 2.3 개정 제안에는 ITC 2001을 옮긴 정지/변화 줄무늬 기준(>5쌍, 355×268 px의 40%/25%)이 Level 3로 들어 있었고, 최종 권고에서는 사라졌다. 삭제 사유를 밝힌 1차 기록은 찾지 못했다.

### Cited Findings
- [원문] WCAG 2.0/2.1/2.2 정의 "general flash and red flash thresholds": "…the combined area of flashes occurring concurrently occupies no more than a total of .006 steradians within any 10 degree visual field on the screen (25% of any 10 degree visual field on the screen) at typical viewing distance … Exception: Flashing that is a fine, balanced, pattern such as white noise or an alternating checkerboard pattern with "squares" smaller than 0.1 degree (of visual field at typical viewing distance) on a side does not violate the thresholds." — [WCAG 2.2](https://www.w3.org/TR/WCAG22/#dfn-general-flash-and-red-flash-thresholds); 동일 문구 [WCAG 2.0](https://www.w3.org/TR/WCAG20/), [WCAG 2.1](https://www.w3.org/TR/WCAG21/)
- [원문] Note 1: "using a 341 x 256 pixel rectangle anywhere on the displayed screen area when the content is viewed at 1024 x 768 pixels will provide a good estimate of a 10 degree visual field for standard screen sizes and viewing distances (e.g., 15-17 inch screen at 22-26 inches)." (2.1/2.2: "This resolution of 75 - 85 ppi is known to be lower, and thus more conservative…") — [WCAG 2.2](https://www.w3.org/TR/WCAG22/)
- [원문] Understanding 2.3.1: "This success criterion is based on existing specifications in use in the UK and by others for television broadcast and has been adapted for computer display viewing… (The 10 degree field is taken from the original specifications and represents the central vision portion of the eye…)"; CSS px 대체 시 341×256 CSS px = 87,296 CSS px; HDR은 "industry standard definition … 20 cd/m2 … below 160 cd/m2. (ITU-R BT.1702.) … Michelson contrast is 1/17 or greater"; WCAG 2.2 신규 Note: 적색 정의 "[ISO 9241-391]". 패턴 관련 서술은 없음 — [Understanding SC 2.3.1 (WCAG 2.2)](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html)
- [2차: 메일 아카이브] 2004-02-17 Gregg Vanderheiden(Trace Center) 제안 "New proposed 2.3": 출처 "ITC Guidance Note for Licensees on Flashing Images and Regular Patterns in Television (revised July 2001)" + "Wisconsin Computer Equivalence Algorithm". Level 3: "Clearly discernible stripes consisting of more than five light-dark pairs in any orientation are not permitted when the stripes are stationary" — 1024×768에서 임의 355×268 px 사각형의 >40%; 변화(방향전환·진동·점멸·대비반전) 줄무늬는 >25%; "clearly discernible" = 어두운 막대 <160 cd/m² & ≥20 cd/m² 차이; 감마 2.2, 피크화이트 200 cd/m² — [w3c-wai-gl 2004JanMar/0306](https://lists.w3.org/Archives/Public/w3c-wai-gl/2004JanMar/0306.html)
- [2차] 2004-03-11 Working Draft에서 2.3이 ITC GN 데이터를 반영해 대폭 재작성되었다는 언급 — 검색 결과 스니펫(같은 메일링리스트) [세부 미확인]
- [2차] 2022-07 Jordan & Vanderheiden "Proposal: Updating WCAG photosensitive epilepsy guidance via notes" — 규범 변경 없이 notes로 WCAG 2.2에 반영 제안 — [w3c-wai-gl 2022JulSep att-0127](https://lists.w3.org/Archives/Public/w3c-wai-gl/2022JulSep/att-0127/WCAG_update_through_notes_draft_20220725.docx)
- [원문] WCAG 3 초안 "No flashing over threshold"(Updated 7 October 2026, [Draft]): 패턴 관련 문구는 동일한 0.1° 예외뿐, "There is research into the size and frequency metrics underway… The exact values are likely to change before WCAG3 is published." — [WCAG 3 draft](https://w3c.github.io/wcag3/informative/animation-and-movement/avoid-physical-harm/no-flashing-over-threshold/)
- [2차] JV24: "Only the WCAG SC has a reference to fine patterns: an exception for white noise or balanced, alternating patterns with blocks of 0.1° (6 arcmin) or smaller. This exception was included in WCAG because there have been no recorded responses to white noise or static on screens at typical viewing distances. No experiments have been done with flashing of patterns that are near the resolution limit of the eye." — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [2차] MDN "Accessibility and spatial patterns"가 EFA 작업그룹의 ">5 light-dark pairs" 문구를 인용하지만 WCAG 규범은 아님 — [MDN](https://developer.mozilla.org/en-US/docs/Web/Accessibility/Guides/Accessibility_and_Spatial_Patterns)

### Inferences
- 0.1° 예외는 "패턴 위험" 기준이 아니라 **플래시 측정에서 미세 패턴을 제외하는 오검출 방지 규정**이다(BT.1702-0 Fig.2의 "block averaging reduces spurious detections due to … fine patterns"와 같은 취지).
- 0.1° 정사각형 체커보드 → 주기 0.2° → 축방향 기본 공간주파수 약 **5 cycles/deg**에 해당(계산 추론, WCAG 원문은 cpd를 쓰지 않음).
- 2004 제안 → 2008 최종 사이 정지 패턴 기준이 사라진 것은, 패턴이 "flash"가 아닌 정적 콘텐츠 속성이라 SC 2.3(Seizures, "flash" 중심)의 범위에서 빠졌을 가능성 — **추정**.

### Gaps
- 패턴 기준 삭제 결정의 회의록·이슈 [미확인]. github.com/w3c/wcag 이슈 중 패턴 SC 제안을 찾지 못함(검색 결과 없음 — 존재하지 않는다고 단정할 수는 없음).

---

## Q6. trace24 (Jordan & Vanderheiden 2024, pseGuidelines)

### Takeaway
JV24는 5대 지침(UK, 일본, ISO, ITU, W3C)을 비교하지만, **제안 provision은 플래시 전용**이고 패턴 조항을 제안하지 않는다. 운동 보상 임계에 관해 "no specific, actionable guidance available"이라고 명시하며, 시거리별 시야각 표(Table 2)를 제시한다.

### Cited Findings
- [원문] 비교 범위: "five major guidelines (from the U.K., Japan, ISO, ITU, and W3C)"; 2004 EFA 워크숍 → 2005 "draft consensus"(Harding et al. 2005), "Wilkins, Emmett, and Harding proposed slightly updated recommendations later that year for regular patterns [109]" — [JV24, ACM TACCESS 17:17 / PMC11872230](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [원문] 6.5.6 Effects of Movement: "Compared to static and oscillating patterns, patterns that drift slowly in one direction result in significantly less synchronized excitation of neurons [16]. The ISO and Ofcom guidelines … are exempt if they exhibit smooth flowing motion "across, into, or out of the screen in one direction" [49, 69]. In Wilkins, Emmett, and Harding [109], drifting patterns are allowed to have a larger number of light-dark pairs of stripes compared to stationary patterns… An example analysis framework given in ITU-R BT.1702 includes motion estimation and motion compensation steps… Documentation for the HardingFPA system [22] also states that motion and noise compensation is performed before flash analysis. Unfortunately, there is no specific, actionable guidance available on motion compensation thresholds that are appropriate for analysis. When is motion so fast that it is perceived as a flicker rather than motion?" — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [원문] Table 2 (시야각·0.006 sr 비율·25% 면적 시 추정 반응률): 20-in TV(4:3) 7H [Binnie 2002] 10.9°×8.2°, 21.2%, 12%; 60-in TV 5H 15.2°×11.4°, 10.9%, 23%; 65-in TV(16:9) 3.2H [BT.2035 참조] 31.0°×17.8°, 4.5%, 42%; 6.1-in 폰 32 cm 24.8°×11.6°, 10.5%, 23%; 19 cm 40.6°×19.4°, 3.8%, 47%; 10.1-in 태블릿(아동) 36.8 cm 32.9°×20.9°, 3.2%, 51%; 29 cm 41.1°×26.4°, 2.0%, 65%; **15-in(4:3) 56 cm [WCAG]** 30.5°×23.1°, 2.7%, 63%; 1920×1080 CSS 기준 71.1 cm 39.3°×22.7°, 2.7%, 56%; **19-in(4:3) 68.0 cm [82 = Shin & Hegde 2010 추정]** 31.0°×24.0°, 2.6%, 57%; 27.5-in 75.7 cm 42.7°×27.5°, 1.9%, 67%; VR 90°×90° 0.3%, 100% — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [원문] "When describing pattern guidelines and rationale, Wilkins, Emmett, and Harding proposed limiting patterns to a combined solid angle of only 0.006 steradians if large, future screens were deemed to be problematic [109]." EFA 인용: "Once 10 degrees of the visual field is affected by flickering or patterned stimuli, further increases in area … has little significance" — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [원문] 면적 임계의 근거: "Area thresholds for the population were set based on experiments done with brightly-lit, high contrast, grating (striped) patterns [14, 110]. Since flashing and pattern photosensitivity have similar effects, it is expected that hazardous flashing areas are similar."; 휘도 임계도 "experiments that used static and phase shifting patterns" 기반 — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [원문] pseGuidelines 제안문: "Proposed Provision (Normative): There are no 1-second time spans, where there are more than 6 luminance transition or red transition counts…" — 정의 항목(critical transition difference 10%/1/17, qualifying duration 66 ms, sufficiently synchronized 20 ms, fast flashes 15 ms, 면적 0.006 sr/25%/416×416 px) 중 **패턴 정의 없음** — [github.com/traceRERC/pseGuidelines](https://github.com/traceRERC/pseGuidelines)
- [원문] 기준 화면: "A 23-inch (diagonal) Full HD monitor viewed at arm's length … 1920 × 1080 px (i.e., 0.0213° CSS pixels) and viewed at 71.1 cm (28 in.)"; "The 25%-of-the-screen metric for television content was chosen for backwards compatibility." — [pseGuidelines](https://github.com/traceRERC/pseGuidelines)

### Inferences
- JV24 본문은 "ISO and Ofcom"만 smooth-flow 면제를 가진다고 쓰는데, BT.1702-3(2023)도 informative Attachment 1에 같은 면제가 있다. JV24가 BT.1702-3을 인용([50])하면서도 패턴 조항을 거론하지 않은 것은 누락 또는 informative 부속서라서 제외한 것으로 보인다(추정).

### Gaps
- Table 2 각주 [82]가 Shin & Hegde 2010인지 본문 참조목록으로 최종 확인 못함(사용자 제공 정보와 일치) — 경미.

---

## Q7. 기타: 호주·캐나다·EBU·미국·게임 업계·Harding FPA

### Takeaway
게임 업계에서는 Xbox XAG 118이 독자적이고 단순화된 공간 패턴 규칙(대비 >10%, 화면 ~20% 이상, 쌍 수 없음)을 공개한다. Harding FPA는 Ofcom 해석을 운영화하여 "≥6 정지 쌍, >0.5 s, >40%, ≥20 cd/m²"를 쓴다. 호주(Free TV)·캐나다·EBU·미국 연방 방송 기준의 패턴 문구는 찾지 못했다.

### Cited Findings
- [원문] Xbox Accessibility Guideline 118 (ms.date 2022-05-09) "Spatial pattern failure — Definition: Alternating bands that have high contrast define a spatial pattern. Failure criteria: The difference in contrast is greater than 10 percent. The pattern takes up a large part of the screen (approximately 20 percent or more). How to address: Reduce the contrast between the bands. Decrease the size of the pattern." 플래시는 "approximately 20 percent or more" 면적, 10% 휘도; 도구로 Harding FPA 링크 — [XAG 118](https://learn.microsoft.com/en-us/gaming/accessibility/xbox-accessibility-guidelines/118)
- [원문: 벤더] Harding FPA 해설: "a stripe consists of 2 opposing spatial changes in luminance"; "pairs of alternating light-dark bars of at least 20cd/m2 contrast"; 체커보드는 무시하고 "arrays of repeated elements which are bar-like in structure"는 검출; 실패 = "6 or more stationary pairs which persist for more than 0.5 seconds" & "the area occupied is greater than 40%"; "A pattern must remain stationary for at least 0.5 seconds in order to fail"; 카메라 팬/줌 운동은 "exempt from failure under the Ofcom guidlines"; "In-depth analysis is averaged over the most recent 5 frames"; 부분 충족 시 "Pass with Caution" — [HardingFPA: How to interpret results](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/)
- [2차] Harding FPA는 2001년부터 상용(BBC 2007 인용) — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [2차] 호주 Network 10/Paramount ANZ 기술 납품 사양에 "Photosensitive Epilepsy" 절 존재, 일반 문구("Programs should not contain sequences of rapid flickering…")만 확인, 패턴 수치 없음 — [Paramount ANZ Technical Delivery Domestic 2026](https://www.paramountanz.com.au/wp-content/uploads/2026/03/10.01.SPEC_Technical-Delivery-Domestic-2026-1.0.pdf) (검색 스니펫)
- [원문] EBU는 ITC 각주에서 overscan 근거(EBU Tech. R95-2000)로만 등장 — [ITC GN](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf)
- [2차] 미국: 연방 방송 규정은 없고 Section 508·EN 301 549가 WCAG SC를 비웹 문서·소프트웨어에 적용 — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)

### Inferences
- XAG 118의 "10% 대비, 20% 면적"은 WCAG의 10% 상대휘도와 TV의 25%를 섞은 독자 단순화로 보이며, ITC 계열의 ">5쌍"·정지/변화 구분·smooth-flow 면제가 모두 빠져 있다.
- Harding FPA의 "≥6쌍"은 ITC ">5쌍"의 정수 표현, "0.5 s 정지 지속"은 표준에 없는 벤더 운영 파라미터.

### Gaps
- Free TV Australia OP(운영 실무 노트)의 플래시/패턴 문구, ACMA, 캐나다 CAB/CRTC, EBU R-권고, Sony/PlayStation TRC(비공개), Nintendo 등 [미확인].
- Harding FPA 변화(진동) 패턴의 25% 기준 처리 방식 [미확인].

---

## Q8. "변화/진동 vs 정지 vs 매끄러운 흐름"의 정량 임계, cpd 정의 여부

### Takeaway
**어떤 공개 표준도** "stationary", "change direction/oscillate", "flow smoothly … in one direction"을 구분하는 정량 임계(속도, 지속시간, 프레임 간 변위)를 정의하지 않으며, **cycles/degree를 정의한 표준도 없다**. 유일한 정량은 벤더(Harding FPA)의 "정지 ≥0.5 s"와 WCAG의 0.1° 미세 패턴 예외(공간 크기)뿐이다.

### Cited Findings
- [원문] ITC/Ofcom/ITU-3 모두 운동 구분은 정성어("stationary", "change direction, oscillate, flash, or reverse in contrast", "obviously flow smoothly across, into, or out of the screen in one direction")만 사용 — [Ofcom Annex 1](https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/broadcast-guidance/programme-guidance/broadcast-code-guidance/section-2-guidance-notes.pdf); [BT.1702-3](https://www.itu.int/rec/R-REC-BT.1702)
- [원문] JV24: "there is no specific, actionable guidance available on motion compensation thresholds" — [JV24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/)
- [원문: 벤더] Harding FPA: "A pattern must remain stationary for at least 0.5 seconds in order to fail" — [HardingFPA](https://www.hardingfpa.com/technical-support/how-to-interpret-hardingfpa-results/)
- [원문] 공간 척도: ITC/Ofcom/ITU는 "pairs" 개수와 화면 면적%만 사용(시거리·cpd 없음, ITU는 "clearly discernible in normal domestic viewing conditions" 전제); WCAG는 "0.1 degree" 및 10° 시야; JV24/pseGuidelines는 0.006 sr·CSS px(0.0213°) — [ITC](https://www.ofcom.org.uk/__data/assets/pdf_file/0021/16248/gn_flash.pdf); [WCAG 2.2](https://www.w3.org/TR/WCAG22/); [pseGuidelines](https://github.com/traceRERC/pseGuidelines)
- [2차] 공간주파수(cpd) 기반 기준은 표준이 아닌 연구 문헌(Wilkins, Emmett & Harding 2005 Epilepsia 46(8):1212–1218 등)에 존재 — [JV24 참고문헌 109](https://pmc.ncbi.nlm.nih.gov/articles/PMC11872230/) (내용은 다른 연구자 담당)

### Inferences
- "쌍 수 + 화면 면적%" 정의는 시거리에 따라 실제 cpd가 달라진다. 예: 화면 높이 H에서 7H 시청 시 세로 시야 ≈8.2°(JV24 Table 2의 20-in TV)이므로 화면 높이 전체에 6쌍이면 ≈0.7 cpd, 동일 패턴을 56 cm 15-in 모니터(세로 23.1°)에서 보면 ≈0.26 cpd — 표준 정의는 공간주파수에 대해 불변이 아님(계산 추론).
- 구현(LUMI 등) 시 "smooth flow" 판정은 표준 근거가 없으므로 자체 작업값(working value)으로 명시하고 문서화해야 한다.

### Gaps
- ISO 9241-391 5.3.1과 NHK·JBA 2020판이 운동/cpd 정량을 포함하는지 [미확인] — 이 두 문서가 유일한 미검증 가능 후보.

---

## 계보 요약 (원천 → 파생)

```
[1993 Pot Noodle 광고 사건]
  └─ ITC Guidance Note 1994-11 (초판, 원문 미확인)
       └─ ITC GN 1999-09 개정 (미확인)
            └─ ITC GN 2001-07 개정 (+2002-06 편집)  ← 패턴 원문 최古 확인본
                 ├─ Ofcom Annex 1 재발행 2005-07-25 → Guidance Notes Issue 12 (2017-07-18): 문구 동일
                 ├─ ITU-R BT.1702-0 (2005-02): 플래시만 복사, 패턴 수치 제외
                 │    ├─ -1 (07/2018), -2 (10/2019, HDR 1/17 추가)
                 │    └─ -3 (11/2023): Guideline 2 + informative Attachment 1 로 ITC/Ofcom 패턴 문구 이식
                 ├─ WCAG 2.0 초안 2004-02 (Level 3 줄무늬 기준) → 최종 2008에서 삭제, 0.1° 예외만 잔존 → 2.1(2018)/2.2(2023)
                 ├─ ISO 9241-391:2016 (IWA 3:2005 경유; 5.3.1 본문 미확인, 2차상 Ofcom 면제 문구 공유)
                 └─ Harding FPA (2001~, Ofcom 운영화: ≥6쌍, >0.5 s, >40%)
[1997-12 포켓몬 사건]
  └─ NHK·민방련 1998-04-08 (독자) → 2006-04-01 (BT.1702-0 참고) → 2020 (미확인)
       패턴: "画面の大部分を占めることも避ける" (무정량)
[독자 계열] Xbox XAG 118 (2022): 대비 >10%, 면적 ~20%
[제안] trace24 (2024): 플래시만, 패턴 없음
```
