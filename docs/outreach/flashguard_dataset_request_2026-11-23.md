# FlashGuard 저자에게 데이터 요청 (초안, 2026-11-23, 사용자 승인)

받는 사람: 논문 저자란의 주소(arXiv:2507.19692 HTML 판 저자 정보) · 보낸 날짜/답장은 아래 기록에 추가.

---

**Subject:** Request for the FlashGuard dataset (arXiv:2507.19692) for academic benchmarking

Dear Ishan Pendyala,

I am a student researcher on LUMI, a university research team in Korea working on automatic detection and mitigation of photosensitive-epilepsy (PSE) triggers in video. We read your paper "FlashGuard" (arXiv:2507.19692) with great interest. Its idea of a sparse trigger array combined with lightweight overlay mitigation is very relevant to our work, and we would like to use FlashGuard as a reference baseline.

The paper states that the dataset is available upon request, so I would like to ask whether you could share:

1. The 1,000 synthetic 10-second videos (or the generation script and parameters) together with the PEAT labels and the 800/200 train/test split.
2. If possible, the code or the fitted parameters that the paper does not report: the logistic-regression coefficients and threshold per node, the interpolation method for the trigger mask, the model that predicts the darkening level k, and the alpha value of the 15-frame average-color overlay.
3. A short clarification of the "92%" mitigation figure: is it the fraction of flash frames removed, or the fraction remaining?

We would use these only for non-commercial academic research: to reimplement FlashGuard faithfully and compare it with our method on the same test set, with proper citation of your work. We will not redistribute the videos or publish any frames, and we handle all flashing material without playback, following our lab's safety rules. We would be glad to share our comparison results with you.

Thank you very much for your time and for making this work available.

Best regards,
[Name]
LUMI research team, [University / Department]

---

## 기록
- 2026-11-23 초안 작성. Gmail 연결이 안 되어 사용자가 직접 발송 예정.
