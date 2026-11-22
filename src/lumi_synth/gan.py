"""조건부 GAN: 오라클과 '같은 물리 모델(선형 빛 더하기)'로, 더 자연스럽고 다양한 깜빡임 빛을 학습한다.

오라클과 GAN 의 관계 (docs/GAN.md §1)
    오라클  X_lin = Y_lin + M · w(t) · δ            빛이 마스크 안에서 '균일'하고 파형이 '정해진 식'(사각·사인·삼각)
    GAN     X_lin = Y_lin + M · Δ_G(t, p | Y, M, c, z)  빛의 공간 분포와 시간 변화를 G 가 장면에 맞게 만든다
    두 방식 모두 정답은 같은 판정 규칙(measure.analyze)이 매긴다. GAN 은 '규칙'을 새로 만들지 않고
    '겉모습'을 다양하게 만든다.

설계와 근거 (실험 기록: 보관 브랜치 docs/GAN_EXPERIMENTS.md)
    1. 오라클을 따라 하게 학습하지 않는다.  E1 에서 오라클 결과를 정답으로 L1 학습하면 G 가 '오라클 평균판'이
       되는 모드 붕괴를 확인했다(같은 조건에서 z 를 바꿔도 세기 표준편차 0.0004). 대신 미분 가능 판정기
       soft_judge 가 '목표 판정(위험 / 문턱 근처 안전)'에 맞추게 한다(E2 에서 목표 조절 91~99% 확인).
    2. 출력은 '빛의 양 Δ ≥ 0' 한 채널, 1/4 해상도.  E2a(RGB 잔차)는 질감이 절반가량 지워졌다. 빛을 더하는
       방식은 원래 윤곽선을 지우지 않고, 1/4 해상도에서 키우면 빛이 부드럽게 퍼진다(조명처럼).
    3. 다양성 손실은 픽셀이 아니라 '빛의 시간 스펙트럼'에.  E2b 에서 픽셀 L1 다양성 손실은 G 가 주파수 대신
       진폭만 바꾸는 쉬운 길을 택하게 했다. 여기서는 마스크 안 평균 빛 Δ(t) 의 주파수 스펙트럼 차이를 키운다.
    4. D 는 프레임 한 장(2D)만 본다.  진짜 = 깨끗한 바탕 프레임. 각 프레임이 자연스러운지(빛 번짐이 어색한
       네모 덩어리가 아닌지)만 판정하고, 깜빡임 자체(프레임 간 변화)는 벌주지 않는다.
    5. 마스크 밖은 구조적으로 원본과 같다(M 을 곱함). 국소성 손실이 필요 없다.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from lumi_synth import measure

TIME_FREQS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)   # 시간 특징: G 가 조합해 깜빡임 리듬을 만든다
DELTA_MAX = 0.6                                      # 더할 수 있는 빛의 최대치(상대 휘도) = 120 cd/m²
TARGET_HAZARD, TARGET_NEAR = 1.0, 0.0                # 조건 c: 위험 / 문턱 근처 안전
# 소프트 판정기 버전. v1(2026-11-09 독립 리뷰 반영): 양방향 동시 후보를 1번으로(D4), 1초 창 최대를 편향 없는
# 최댓값으로. v0 은 반대 위상 깜빡임을 2번으로 세고(2 Hz 안전 영상이 8.8), log-sum-exp 로 약 +0.8 과대였다.
SOFT_JUDGE_VERSION = "v1"


# ============================================================================================
# 1. 색 공간 (oracle.py 와 같은 IEC 61966-2-1 식, torch 판)
# ============================================================================================
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def luminance_rel(lin: torch.Tensor) -> torch.Tensor:
    """[B,3,T,H,W] 선형 RGB(채널 순서 B,G,R — OpenCV 와 같음) → 상대 휘도 [B,T,H,W]."""
    return 0.0722 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.2126 * lin[:, 2]


# ============================================================================================
# 2. 미분 가능 판정기 (soft judge) — measure.analyze 의 각 단계를 시그모이드로 완화
# ============================================================================================
def soft_judge(y: torch.Tensor, fps: float, *, tau_l: float = 0.02, tau_a: float = 0.03) -> dict:
    """상대 휘도 y [B,T,H,W] → 1초 창 최대 화면 전이 수의 연속 근사 (G 학습용).

    하드 판정과의 대응 (차이는 docs/GAN.md §3 에 정리, 학습 후 하드 판정으로 반드시 다시 잰다)
      픽셀 전이   : 하드는 '직전 극값 대비 ±20 cd/m²'(히스테리시스). 소프트는 '1~c 프레임 전 대비' 변화의
                   큰 쪽으로 근사(c = 동시성 창). → 아주 느린 변화(c 프레임 넘게 걸친 변화)는 덜 센다.
      어두운 쪽   : min(현재, 이전) < 160 cd/m² (0.8) 를 시그모이드로.
      면적        : 방향별 평균 > 25% 를 시그모이드로 (D2 와 같음).
      상태 기계   : '직전 프레임에 같은 방향 후보가 없었을 때만 새 전이'로 근사 (D1 의 근사).
                   두 방향이 같은 프레임에 모두 새 후보면 1번(D4): 두 방향의 새 전이 중 큰 값만 센다.
      1초 창 최대 : 이동 합(conv1d) 후 최댓값(amax). 기울기는 최대 창으로만 흐른다(부분 기울기).
                   log-sum-exp 는 창이 여러 개 같은 값일 때 약 +0.8 과대 추정해서 쓰지 않는다.
    """
    d_thr = measure.DELTA_CD / measure.WHITE_CD          # 0.1
    dark_thr = measure.DARK_LIMIT_CD / measure.WHITE_CD  # 0.8
    c = measure.concurrency_frames(fps)
    ups, dns = [], []
    for k in range(1, c + 1):                            # k 프레임 전과 비교
        cur, prev = y[:, k:], y[:, :-k]
        d = cur - prev
        dark = torch.sigmoid((dark_thr - torch.minimum(cur, prev)) / tau_l)
        up = torch.sigmoid((d - d_thr) / tau_l) * dark
        dn = torch.sigmoid((-d - d_thr) / tau_l) * dark
        pad = (0, 0, 0, 0, k, 0)                         # 앞쪽 k 프레임을 0으로 채워 길이 T 로 맞춤
        ups.append(F.pad(up, pad))
        dns.append(F.pad(dn, pad))
    up = torch.stack(ups).amax(0)                        # [B,T,H,W]
    dn = torch.stack(dns).amax(0)
    return _screen_count(up, dn, fps, tau_a)


def _screen_count(up: torch.Tensor, dn: torch.Tensor, fps: float, tau_a: float) -> dict:
    """픽셀별 전이 점수(up·dn [B,T,H,W], 0~1) → 방향별 면적 → 새 화면 전이 → 1초 창 최대 전이 수.

    휘도(soft_judge)와 적색(soft_judge_red)이 같은 면적·횟수 규칙을 쓰므로 함께 쓴다(measure.flash_from_transitions 와 같은 관계)."""
    a_up, a_dn = up.mean(dim=(2, 3)), dn.mean(dim=(2, 3))  # 방향별 면적 [B,T]
    e_up = torch.sigmoid((a_up - measure.AREA_LIMIT) / tau_a)
    e_dn = torch.sigmoid((a_dn - measure.AREA_LIMIT) / tau_a)
    new_up = e_up * (1 - F.pad(e_up[:, :-1], (1, 0)))
    new_dn = e_dn * (1 - F.pad(e_dn[:, :-1], (1, 0)))
    onset = torch.maximum(new_up, new_dn)                # D4: 같은 프레임 두 방향 = 1번
    w = measure.window_frames(fps)                       # 판정기와 같은 1초 창 (ceil)
    kernel = torch.ones(1, 1, w, device=up.device, dtype=onset.dtype)
    counts = F.conv1d(onset.unsqueeze(1), kernel, padding=w - 1).squeeze(1)
    soft_max = counts.amax(dim=1)
    return {"count": soft_max, "onset": onset, "area_up": a_up, "area_down": a_dn}


# 적색 소프트 판정기 (2026-11-14) — measure.red_state · red_transitions 의 각 단계를 시그모이드로 완화
#   포화 빨강  : 선형 R/(R+G+B) ≥ 0.8 이고 R−G−B ≥ 0.0625        → 두 시그모이드의 곱
#   색 차이    : 바로 앞 프레임과 Δu′v′ > 0.2                      → 시그모이드
#   적색 전이  : (앞이 빨강 아님 · 지금 빨강 · 색 차이) = 빨강으로, 반대 = 빨강에서
#   이후는 휘도와 같은 _screen_count (방향별 면적 > 25%, 1초 창 최대)
# 문턱에 여유(RED_SOFT_MARGIN)를 둬 소프트에서 겨우 넘는 값이 하드에서 떨어지지 않게 한다. 하드 판정의
# '깜빡임 쌍 짓기'(반대 방향 변화 한 쌍만 셈)는 흉내 내지 않는다 → 학습 뒤 반드시 하드 판정으로 다시 잰다.
RED_SOFT_MARGIN = (0.03, 0.02, 0.01)   # 포화 비율, R−G−B, Δu′v′ 에 더하는 여유
_XYZ_BGR = ((0.1804375, 0.3575761, 0.4124564),          # X = 이 행 · (B, G, R)   (measure._XYZ 와 같은 sRGB D65 행렬)
            (0.0721750, 0.7151522, 0.2126729),          # Y
            (0.9503041, 0.1191920, 0.0193339))          # Z


def red_state_soft(lin: torch.Tensor, tau_s: float = 0.05, tau_e: float = 0.02):
    """선형 BGR [B,3,T,H,W] → (포화 빨강 점수 [B,T,H,W] 0~1, u′, v′)."""
    b, g, r = lin[:, 0], lin[:, 1], lin[:, 2]
    ratio = r / (r + g + b).clamp(min=1e-6)
    sat = (torch.sigmoid((ratio - measure.RED_SAT_RATIO - RED_SOFT_MARGIN[0]) / tau_s)
           * torch.sigmoid((r - g - b - measure.RED_MIN_EXCESS - RED_SOFT_MARGIN[1]) / tau_e))
    x, y, z = (m[0] * b + m[1] * g + m[2] * r for m in _XYZ_BGR)
    d = (x + 15.0 * y + 3.0 * z).clamp(min=1e-6)
    ok = y >= measure.RED_BLACK_Y                         # 거의 검정은 색도를 D65 로 고정 (measure 와 같음)
    u = torch.where(ok, 4.0 * x / d, torch.full_like(x, measure.D65_UV[0]))
    v = torch.where(ok, 9.0 * y / d, torch.full_like(x, measure.D65_UV[1]))
    return sat, u, v


def soft_judge_red(lin: torch.Tensor, fps: float, *, tau_uv: float = 0.02, tau_a: float = 0.03) -> dict:
    """선형 BGR [B,3,T,H,W] → 1초 창 최대 적색 화면 전이 수의 연속 근사 (적색 G 학습용)."""
    sat, u, v = red_state_soft(lin)
    duv = torch.sqrt((u[:, 1:] - u[:, :-1]) ** 2 + (v[:, 1:] - v[:, :-1]) ** 2 + 1e-12)
    big = torch.sigmoid((duv - measure.RED_DELTA_UV - RED_SOFT_MARGIN[2]) / tau_uv)
    pad = (0, 0, 0, 0, 1, 0)                              # 첫 프레임은 비교할 앞 프레임이 없다
    to_red = F.pad(sat[:, 1:] * (1 - sat[:, :-1]) * big, pad)
    from_red = F.pad(sat[:, :-1] * (1 - sat[:, 1:]) * big, pad)
    return _screen_count(to_red, from_red, fps, tau_a)


# ============================================================================================
# 3. 생성기 G
# ============================================================================================
def time_features(frames: int, fps: float, device=None) -> torch.Tensor:
    """[2·len(TIME_FREQS), T] 사인·코사인. 합성곱만으로는 '1초에 몇 번'을 세기 어려워 시계를 넣어 준다."""
    t = torch.arange(frames, device=device, dtype=torch.float32) / fps
    return torch.stack([f(2 * math.pi * hz * t) for hz in TIME_FREQS for f in (torch.sin, torch.cos)])


class FiLMBlock(nn.Module):
    """Conv3d → GroupNorm → FiLM(조건) → LeakyReLU.

    FiLM: 조건 벡터(z, c)로 채널마다 배율 γ 와 이동 β 를 만들어 특징에 곱하고 더한다.
    z 가 클립 전체에 같은 γ, β 를 주므로 '이 클립은 이런 리듬·세기로 깜빡인다'는 전역 선택이 가능하다
    (E1 의 복셀별 잡음은 클립 전체에 일관된 선택을 못 했다).
    """

    def __init__(self, cin, cout, cond, kernel=(3, 3, 3), dilation=(1, 1, 1), stride=(1, 1, 1)):
        super().__init__()
        pad = tuple(d * (k // 2) for k, d in zip(kernel, dilation))
        self.conv = nn.Conv3d(cin, cout, kernel, stride=stride, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.film = nn.Linear(cond, 2 * cout)

    def forward(self, x, cond):
        x = self.norm(self.conv(x))
        g, b = self.film(cond).chunk(2, dim=1)
        return F.leaky_relu(x * (1 + g[:, :, None, None, None]) + b[:, :, None, None, None], 0.2)


class LightGenerator(nn.Module):
    """입력 Y [B,3,T,H,W](0~1 sRGB, BGR 순), M [B,1,1,H,W], c [B], z [B,zdim] → X [B,3,T,H,W].

    구조: 부호화(공간 1/4) → 시간 팽창 합성곱(수용 영역 > 1초) → 빛 Δ 1채널 → 원래 크기로 부드럽게 확대
          → X_lin = Y_lin + M·Δ  → sRGB
    """

    hazard = "lum"
    smooth = False      # True (2026-11-15): z 의 마지막 성분으로 시간 흐림 σ ∈ [0, smooth_max] 프레임을 정해 Δ 에 건다
    smooth_max = 2.0    # 휘도 v2: SMOOTH_SIGMA_MAX. 줄무늬 v3: 12 프레임(덮는 정도가 거의 일정 = 진짜 정지 줄무늬)
    smooth_radius = 6

    def __init__(self, width: int = 16, zdim: int = 8, fps: float = 24.0):
        super().__init__()
        self.zdim, self.fps = zdim, fps
        cond = zdim + 1
        cin = 3 + 1 + 1 + 2 * len(TIME_FREQS)
        w = width
        self.e1 = FiLMBlock(cin, w, cond, stride=(1, 2, 2))            # 1/2
        self.e2 = FiLMBlock(w, 2 * w, cond, stride=(1, 2, 2))          # 1/4
        # 시간 팽창 2·4·8: 3×3 커널로 수용 영역 1+2(2+4+8)=29 프레임 ≈ 1.2초 @24fps → '1초 창'을 볼 수 있음
        self.t1 = FiLMBlock(2 * w, 2 * w, cond, kernel=(3, 1, 1), dilation=(2, 1, 1))
        self.t2 = FiLMBlock(2 * w, 2 * w, cond, kernel=(3, 1, 1), dilation=(4, 1, 1))
        self.t3 = FiLMBlock(2 * w, 2 * w, cond, kernel=(3, 1, 1), dilation=(8, 1, 1))
        self.s1 = FiLMBlock(2 * w, 2 * w, cond)                        # 공간 정리
        self.head = nn.Conv3d(2 * w, 1, 3, padding=1)

    def light(self, clean, mask, target, z):
        """빛의 양 Δ [B,1,T,H,W] ∈ [0, DELTA_MAX] (마스크 곱하기 전)."""
        b, _, t, h, w = clean.shape
        cond = torch.cat([z, target[:, None]], dim=1)
        tf = time_features(t, self.fps, clean.device)[None, :, :, None, None].expand(b, -1, t, h, w)
        x = torch.cat([clean, mask.expand(b, 1, t, h, w), target[:, None, None, None, None].expand(b, 1, t, h, w), tf], 1)
        x = self.e2(self.e1(x, cond), cond)
        x = self.s1(self.t3(self.t2(self.t1(x, cond), cond), cond), cond)
        delta = DELTA_MAX * torch.sigmoid(self.head(x))
        if self.smooth:
            delta = temporal_blur(delta, self.smooth_max * 0.5 * (1 + torch.erf(z[:, -1] / math.sqrt(2))), self.smooth_radius)
        return F.interpolate(delta, size=(t, h, w), mode="trilinear", align_corners=False)

    def forward(self, clean, mask, target, z):
        delta = self.light(clean, mask, target, z)
        lin = srgb_to_linear(clean) + mask * delta          # 세 채널에 같은 빛 → 휘도가 정확히 Δ 만큼 오름
        out = linear_to_srgb(lin)
        return clean + mask * (out - clean), delta          # 마스크 밖은 정확히 원본


SMOOTH_SIGMA_MAX = 2.0     # 휘도 G 의 시간 흐림 상한(프레임). 24 fps 에서 σ 2 프레임 = 83 ms


def temporal_blur(x: torch.Tensor, sigma: torch.Tensor, radius: int = 6) -> torch.Tensor:
    """[B,C,T,H,W] 를 시간 축으로 표본마다 다른 가우시안 σ(프레임, [B])로 흐린다. σ → 0 이면 그대로. 끝은 복제.

    도입 이유(2026-11-15): 휘도 G(w_div2.0_2k_s0)의 위험 깜빡임은 모두 한두 프레임에 바뀌는 사각파꼴이었다
    (큰 프레임 변화 / 전체 진폭 0.92~1.00, 규칙 사인·삼각파는 0.36~0.85). 이 데이터로 학습한 탐지기가 규칙 생성기의
    사인·삼각파 휘도 깜빡임을 놓쳤다(v4_roi_b·c). z 로 흐림 정도를 무작위로 주면 G 는 그 파형에서 목표를 맞추는 법을 배운다."""
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    s = sigma.to(x.dtype).clamp(min=1e-3)[:, None]
    w = torch.exp(-0.5 * (k[None] / s) ** 2)
    w = w / w.sum(1, keepdim=True)                                       # [B,K]
    b, c, t, h, wd = x.shape
    xp = F.pad(x.permute(0, 1, 3, 4, 2).reshape(b, c * h * wd, t), (radius, radius), mode="replicate")
    win = xp.unfold(2, 2 * radius + 1, 1)                                # [B, CHW, T, K]
    y = (win * w[:, None, None, :]).sum(-1)
    return y.reshape(b, c, h, wd, t).permute(0, 1, 4, 2, 3)


# ============================================================================================
# 3-2. 적색·줄무늬 생성기 (2026-11-14)
#
# 휘도 G 와 같은 원리를 종류별로 쓴다: ① 물리적으로 섞는 식을 정해 두고 G 는 '얼마나'만 정한다 ② 미분 가능한
# 판정으로 목표(위험 / 근처 안전)를 맞춘다 ③ 학습 뒤 하드 판정으로 다시 잰다 ④ 마스크 밖은 원본 그대로.
# 생성기는 종류마다 따로 학습한다(체크포인트 config.hazard). 같은 신경망 몸통(LightGenerator)을 쓰고 섞는 식만 다르다.
#
#   적색   X_lin = (1 − M·ρ)·Y_lin + M·ρ·R(Y),  R(Y) = 같은 휘도의 순수 빨강 (선형 R = Y/0.2126, [RED_FLOOR, 1] 로 자름)
#          ρ(t,p) ∈ [0,1] = G 가 정하는 '빨강으로 바꾸는 정도'. 휘도를 지키므로(선형 R ≤ 1 인 한) 휘도 깜빡임 없이 색만 바뀐다.
#          밝은 픽셀(Y > 0.2126 = 42.5 cd/m²)은 순수 빨강이 그 밝기를 못 내 어두워진다 → 휘도 전이도 함께 생길 수 있다.
#   줄무늬 X_lin = (1 − M·a)·Y_lin + M·a·S,  S = 사인 줄무늬(어두운 줄 STRIPE_LO ↔ 밝은 줄 hi)
#          a(t,p) ∈ [0,1] = G 가 정하는 '줄무늬가 덮는 정도'(어디에·언제·얼마나 진하게).
#          클립마다 공간 주파수 cpd·방향 θ·흐름 속도·밝은 줄 휘도 hi 를 조건(z, 목표)에서 뽑는다.
#          첫 버전은 직선 줄무늬만 만든다(동심원·방사형·물결 없음, 명암 반전 없음 — 정지와 한 방향 흐름).
#          판정 대리: a·(hi − lo) 로 명암이 기준을 넘는 픽셀의 면적 > 40% 인 프레임이 1초 창 안에 얼마나 이어지는지.
#          줄무늬 식을 우리가 정했으므로 cpd 대역·규칙성은 구조상 지켜진다. 이 대리 판정은 영상 픽셀을 보지 않으므로
#          반드시 하드 판정(measure.analyze_pattern 또는 stripe_freq)으로 다시 잰다.
# ============================================================================================
HAZARDS = ("lum", "red", "stripe")
RED_FLOOR = 0.10                    # 빨강 목표색의 최소 선형 R (포화 빨강 하한 0.0625 에 여유)
STRIPE_CPD = (1.3, 7.0)             # 만드는 공간 주파수 범위 (판정 대역 1~10 cpd 안쪽. 7 cpd = 768 px 화면에서 주기 3.3 px)
STRIPE_LO = 0.05                    # 어두운 줄 상대 휘도 (10 cd/m²)
STRIPE_HI = (0.40, 0.70)            # 밝은 줄 상대 휘도 범위 (80~140 cd/m²)
STRIPE_SPEED_MAX = 6.0              # 흐름 속도 상한 (°/s)
STRIPE_CONTRAST_MIN = 0.15          # 줄무늬로 칠 명암 a·(hi − lo) 하한 (30 cd/m² — 기준 20 에 여유)
STRIPE_HAZARD_SEC = 0.7             # 위험 목표: 1초 창 안 후보 지속 ≥ 0.7초 (기준 0.5초에 여유)
STRIPE_NEAR_SEC = 0.25              # 근처 안전 목표: 지속 ≤ 0.25초
STRIPE_NEAR_AREA = 0.20             # 근처 안전에서도 줄무늬는 보여야 한다: 최대 면적 ≥ 20%
STRIPE_LOSS_SCALE = 10.0            # 지속(초) 손실을 전이 수 손실과 비슷한 크기로


class RedGenerator(LightGenerator):
    """적색 깜빡임: 몸통은 LightGenerator, 출력 ρ ∈ [0,1] 로 '빨강 아닌 상태 O' 와 '빨강 R' 을 섞는다.

    palette=False (첫 버전, r_div2.0_s0·r_crisp2_div2.0_s0): O = 장면 그대로, R = 장면과 같은 휘도의 순수 빨강.
    palette=True (2026-11-14 v2): 클립마다 작은 MLP 가 (z, 목표)에서 색 조건 9개를 뽑는다(red_palette 참고).
      첫 버전으로 학습한 탐지기가 규칙 생성기의 '평평한 빨강 ↔ 평평한 보라' 적색을 24개 중 1개만 찾았다
      (checks/gdr/roi_a_val_rule.md) — 빨강 한 가지 모양만 배웠기 때문이다. v2 는 빨강의 채도·밝기·평평함과
      빨강 아닌 상태의 색·물듦 정도·평평함을 z 가 고르게 해서, 첫 버전(장면 ↔ 장면 밝기의 빨강)과 규칙 생성기
      (평평한 두 색 교대)를 모두 그 안에 담는다.
    palette="z" (v3): 색 조건을 배우지 않고 z 의 앞 9개 성분을 정규 누적분포 Φ 로 0~1 균등값으로 바꿔 그대로 쓴다
      (도메인 무작위화). v2 에서 배운 색 조건이 가운데에 몰렸다(표준편차 0.05~0.09, 물듦 0.31~0.71 —
      평평한 두 색 교대 κ≈1 이나 첫 버전 κ≈0 에 닿지 못함). G 는 FiLM 으로 z 를 받으므로 주어진 색에 맞춰
      깜빡임만 조절한다. zdim ≥ 9 필요.
    palette="z2" (v4): v3 + 10번째 조건 '등휘도' ι (빨강의 휘도를 빨강 아닌 상태의 휘도 쪽으로 당김, 1 = 같은 휘도),
      그리고 모든 조건을 양 끝까지 늘려(0~1 균등 → [−0.2, 1.2] 를 자름) 0·1 이 약 14% 씩 나오게 한다.
      v3 데이터로 학습한 탐지기는 규칙 생성기 적색(밝기 변화 0.02~0.07 cd/m², 무늬 없음)을 24개 중 4개만 찾았다.
      v3 적색은 두 상태의 밝기가 늘 4~18 cd/m² 달라서 탐지기가 '색과 함께 바뀌는 밝기'에 기댄 것으로 본다.
      zdim ≥ 10 필요."""
    hazard = "red"

    def __init__(self, width: int = 16, zdim: int = 8, fps: float = 24.0, palette: bool | str = False):
        super().__init__(width, zdim, fps)
        self.palette = palette
        self.n_par = RED_PALETTE_N + (1 if palette == "z2" else 0)
        if palette in ("z", "z2"):
            if zdim < self.n_par:
                raise ValueError(f"palette='{palette}' 는 zdim ≥ {self.n_par} 이 필요합니다")
        elif palette:
            self.par = nn.Sequential(nn.Linear(zdim + 1, 32), nn.LeakyReLU(0.2), nn.Linear(32, RED_PALETTE_N))

    def palette_params(self, target, z) -> torch.Tensor:
        """[B, RED_PALETTE_N] 모두 0~1 (뜻은 red_palette 참고)."""
        if self.palette in ("z", "z2"):
            u = 0.5 * (1 + torch.erf(z[:, :self.n_par] / math.sqrt(2)))
            return u if self.palette == "z" else (1.4 * u - 0.2).clamp(0.0, 1.0)
        return torch.sigmoid(self.par(torch.cat([z, target[:, None]], dim=1)))

    def forward(self, clean, mask, target, z):
        rho = self.light(clean, mask, target, z) / DELTA_MAX
        lin = srgb_to_linear(clean)
        a = mask * rho
        if not self.palette:
            r_t = (luminance_rel(lin) / 0.2126).clamp(RED_FLOOR, 1.0)[:, None]          # [B,1,T,H,W]
            red = torch.cat([torch.zeros_like(r_t), torch.zeros_like(r_t), r_t], dim=1)  # BGR
            out = linear_to_srgb((1 - a) * lin + a * red)
            return clean + mask * (out - clean), rho
        par = self.palette_params(target, z)
        other, red = red_palette(lin, par)
        out = linear_to_srgb((1 - mask) * lin + mask * ((1 - rho) * other + rho * red))
        return clean + mask * (out - clean), rho, par


RED_PALETTE_N = 9
RED_SAT_RANGE = (0.84, 0.98)        # 빨강 R/(R+G+B) — 포화 기준 0.8 에 여유
RED_GAIN_RANGE = (0.6, 1.6)         # 장면을 따르는 빨강: 장면 휘도의 배율
RED_FLAT_RANGE = (0.02, 0.25)       # 평평한 빨강의 상대 휘도 (4~50 cd/m²)
OTHER_FLAT_RANGE = (0.02, 0.45)     # 평평한 '빨강 아닌 색'의 상대 휘도
OTHER_R_MAX = 0.5                   # '빨강 아닌 색' 의 빨강 성분 상한(색 방향 가중치)


def red_palette(lin: torch.Tensor, par: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """선형 BGR [B,3,T,H,W] + 색 조건 par [B,9](0~1) → (빨강 아닌 상태 O, 빨강 R), 둘 다 [B,3,T,H,W] 선형.

    par: 0 채도 s · 1 빨강 평평함 f_r · 2 빨강 밝기 q_r · 3 물듦 κ · 4~6 O 의 색 방향(B,G,R) · 7 O 평평함 f_o · 8 O 밝기 q_o
      빨강:  목표 휘도 Y_r = (1 − f_r)·gain(q_r)·Y + f_r·flat(q_r),  R = (g, g, r) 로 R/(R+G+B) = s, 휘도 = Y_r
      O:    색 방향 w(빨강 성분 ≤ 0.5)로 휘도 Y_o = (1 − f_o)·Y + f_o·flat(q_o) 인 색 C,  O = (1 − κ)·장면 + κ·C
    κ = 0, f_r = 0, gain = 1, s → 1 이면 첫 버전과 같고, κ = 1·평평함 1 이면 규칙 생성기처럼 두 평평한 색이 교대한다."""
    p = par[:, :, None, None, None, None]                                           # [B,9,1,1,1,1]
    y = luminance_rel(lin)[:, None]                                                 # [B,1,T,H,W]

    def lerp(lo_hi, q):
        return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * q

    s = lerp(RED_SAT_RANGE, p[:, 0])
    y_r = (1 - p[:, 1]) * lerp(RED_GAIN_RANGE, p[:, 2]) * y + p[:, 1] * lerp(RED_FLAT_RANGE, p[:, 2])
    y_o = (1 - p[:, 7]) * y + p[:, 7] * lerp(OTHER_FLAT_RANGE, p[:, 8])
    if par.shape[1] > RED_PALETTE_N:                                                # v4: 등휘도 ι
        y_r = (1 - p[:, 9]) * y_r + p[:, 9] * y_o
    k = (1 - s) / (2 * s)                                                           # G = B = k·R
    r = (y_r / (0.2126 + 0.7874 * k)).clamp(RED_FLOOR, 1.0)
    red = torch.cat([k * r, k * r, r], dim=1)
    wdir = torch.cat([p[:, 4], p[:, 5], OTHER_R_MAX * p[:, 6]], dim=1) + 1e-3      # BGR 가중치
    lum_w = 0.0722 * wdir[:, 0:1] + 0.7152 * wdir[:, 1:2] + 0.2126 * wdir[:, 2:3]
    color = (wdir * y_o / lum_w).clamp(0.0, 1.0)
    other = (1 - p[:, 3]) * lin + p[:, 3] * color
    return other, red


def palette_diversity(par1: torch.Tensor, par2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """적색 v2 다양성: z 가 다르면 색 조건(0~1 9개)이 달라지도록(음수로 최소화)."""
    gap = (par1 - par2).abs().mean(1)
    return -(gap / ((z1 - z2).abs().mean(1) + 1e-6)).mean()


class StripeGenerator(LightGenerator):
    """줄무늬: 몸통은 LightGenerator(출력 a = 덮는 정도), 클립별 줄무늬 조건은 작은 MLP 가 (z, 목표)에서 뽑는다.

    cond="z" (v2, 2026-11-15, 2차 — 반전·흐름 줄무늬): 줄무늬 조건을 배우지 않고 z 의 앞 8개 성분을 Φ 로 0~1 균등값으로
      바꿔 정한다(적색 v2→v3 의 교훈: 배운 조건은 가운데로 몰린다). 조건 8개 → par [B,8]:
        cpd(로그 균등 1.3~7) · 방향 θ ∈ [0, 2π) · 움직임(정지 / 명암 반전 / 흐름, 각 1/3) · 흐름 속도 1~6 °/s ·
        반전 주파수 2~10 Hz · 밝은 줄 휘도 · 줄 모양(사인 ↔ 사각) · 반전 파형(코사인 ↔ 사각)
      G 는 덮는 정도 a(t,p) 만 정한다. zdim ≥ 8 필요."""
    hazard = "stripe"
    cond = "mlp"
    decouple = False    # v4(2026-11-16): z[8] < 0 인 클립(절반)은 픽셀의 시간 주파수를 DECOUPLE_HZ 이하로 — 휘도 위험 없는 반전·흐름
    patterns = False    # v5(2026-11-16): 직선 외 무늬(동심원·방사형·나선·처프·체커·물결) — pattern_profile

    def __init__(self, width: int = 16, zdim: int = 8, fps: float = 24.0, cond: str = "mlp"):
        super().__init__(width, zdim, fps)
        self.cond = cond
        if cond == "z":
            if zdim < 8:
                raise ValueError("cond='z' 는 zdim ≥ 8 이 필요합니다")
        else:
            self.par = nn.Sequential(nn.Linear(zdim + 1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 5))

    def stripe_params_z(self, z, ppd: float) -> torch.Tensor:
        """v2: [B,8] = (cpd, θ, 흐름 속도 °/s(정지·반전이면 0), 밝은 줄 상대 휘도, 반전 Hz(반전 아니면 0), 줄 모양 g, 반전 파형 g, 움직임 0/1/2)."""
        u = 0.5 * (1 + torch.erf(z[:, :8] / math.sqrt(2)))
        hi_cpd = min(STRIPE_CPD[1], ppd / 3.0)
        lo_cpd = min(STRIPE_CPD[0], hi_cpd / 1.5)
        cpd = lo_cpd * (hi_cpd / lo_cpd) ** u[:, 0]
        theta = 2 * math.pi * u[:, 1]
        motion = torch.clamp((u[:, 2] * 3).floor(), max=2)                     # 0 정지 · 1 반전 · 2 흐름
        speed = torch.where(motion == 2, 1.0 + (STRIPE_SPEED_MAX - 1.0) * u[:, 3], torch.zeros_like(cpd))
        rev_hz = torch.where(motion == 1, STRIPE_REV_HZ[0] + (STRIPE_REV_HZ[1] - STRIPE_REV_HZ[0]) * u[:, 4], torch.zeros_like(cpd))
        hi = STRIPE_HI[0] + (STRIPE_HI[1] - STRIPE_HI[0]) * u[:, 5]
        shape_g = 0.5 + 9.5 * u[:, 6]                                          # tanh 날카로움: 0.5 ≈ 사인, 10 ≈ 사각
        rev_g = 0.5 + 9.5 * u[:, 7]
        dc = torch.zeros_like(cpd)
        if self.decouple and z.shape[1] > 8:
            # 줄무늬 위험과 휘도 위험 떼어 놓기: 반전·흐름 줄무늬는 각 픽셀이 (반전 Hz) 또는 (속도 × cpd) Hz 로 깜빡여
            # 휘도 위험(초당 3회 초과)이 함께 생긴다. v7 학습 데이터에서 줄무늬 위험 표본의 대부분이 휘도 위험도 가졌고, 탐지기가
            # 규칙 생성기의 휘도 깜빡임 24개 중 21개를 줄무늬로 잡았다. 절반의 클립은 픽셀 깜빡임을 DECOUPLE_HZ 이하로 묶는다.
            dc = (z[:, 8] < 0).float()
            rev_hz = torch.where((dc > 0) & (motion == 1), 0.5 + (DECOUPLE_HZ - 0.5) * u[:, 4], rev_hz)
            speed = torch.where((dc > 0) & (motion == 2), torch.minimum(speed, DECOUPLE_HZ / cpd), speed)
        if self.patterns and z.shape[1] >= 14:
            # v5(2026-11-16): 무늬 종류 z[9] (PATTERN_SHAPES 중 균등), 중심 z[10]·z[11] (화면의 25~75%), 추가 값 z[12]
            kind = torch.clamp((0.5 * (1 + torch.erf(z[:, 9] / math.sqrt(2))) * len(PATTERN_SHAPES)).floor(), max=len(PATTERN_SHAPES) - 1)
            cx = 0.25 + 0.5 * 0.5 * (1 + torch.erf(z[:, 10] / math.sqrt(2)))
            cy = 0.25 + 0.5 * 0.5 * (1 + torch.erf(z[:, 11] / math.sqrt(2)))
            extra = 0.5 * (1 + torch.erf(z[:, 12] / math.sqrt(2)))
            return torch.stack([cpd, theta, speed, hi, rev_hz, shape_g, rev_g, motion, dc, kind, cx, cy, extra], dim=1)
        return torch.stack([cpd, theta, speed, hi, rev_hz, shape_g, rev_g, motion, dc], dim=1)

    def stripe_params(self, target, z, ppd: float) -> torch.Tensor:
        """[B,4] = (cpd, θ 라디안 [0, π), 흐름 속도 °/s, 밝은 줄 상대 휘도). cond='z' 면 [B,8](stripe_params_z)."""
        if self.cond == "z":
            return self.stripe_params_z(z, ppd)
        p = self.par(torch.cat([z, target[:, None]], dim=1))
        hi_cpd = min(STRIPE_CPD[1], ppd / 3.0)                 # 주기가 3 px 아래로 내려가지 않게(작은 해상도 동작 확인용)
        lo_cpd = min(STRIPE_CPD[0], hi_cpd / 1.5)
        cpd = lo_cpd * (hi_cpd / lo_cpd) ** torch.sigmoid(p[:, 0])
        theta = 0.5 * torch.atan2(p[:, 2], p[:, 1] + 1e-6) % math.pi
        v = torch.tanh(p[:, 3])                                # |v| < 0.3 은 정지 (속도 0)
        speed = STRIPE_SPEED_MAX * torch.sign(v) * F.relu(v.abs() - 0.3) / 0.7
        hi = STRIPE_HI[0] + (STRIPE_HI[1] - STRIPE_HI[0]) * torch.sigmoid(p[:, 4])
        return torch.stack([cpd, theta, speed, hi], dim=1)

    def forward(self, clean, mask, target, z):
        b, _, t, h, w = clean.shape
        amp = self.light(clean, mask, target, z) / DELTA_MAX   # [B,1,T,H,W]
        ppd = measure.pixels_per_degree(w, h)
        par = self.stripe_params(target, z, ppd)
        cpd, theta, speed, hi = par[:, :4].unbind(1)
        ys = torch.arange(h, device=clean.device, dtype=torch.float32) - (h - 1) / 2
        xs = torch.arange(w, device=clean.device, dtype=torch.float32) - (w - 1) / 2
        q = (xs[None, None, :] * torch.cos(theta)[:, None, None]
             + ys[None, :, None] * torch.sin(theta)[:, None, None])                  # [B,H,W] 줄 수직 방향 좌표(px)
        shift = speed[:, None] * ppd * torch.arange(t, device=clean.device)[None] / self.fps   # [B,T] 흐른 거리(px)
        phase = (2 * math.pi * cpd / ppd)[:, None, None, None] * (q[:, None] - shift[:, :, None, None])
        if par.shape[1] >= 13:                                 # v5: 무늬 종류별
            prof = torch.stack([pattern_profile(par[i], h, w, t, self.fps, ppd) for i in range(b)])
            prof = prof * stripe_reversal(par, t, self.fps)[:, :, None, None]
            stripe = STRIPE_LO + (hi[:, None, None, None] - STRIPE_LO) * (0.5 + 0.5 * prof)
        elif self.cond == "z":                                 # v2: 줄 모양·명암 반전
            g = par[:, 5][:, None, None, None]
            prof = torch.tanh(g * torch.sin(phase)) / torch.tanh(g)
            prof = prof * stripe_reversal(par, t, self.fps)[:, :, None, None]
            stripe = STRIPE_LO + (hi[:, None, None, None] - STRIPE_LO) * (0.5 + 0.5 * prof)
        else:
            stripe = STRIPE_LO + (hi[:, None, None, None] - STRIPE_LO) * (0.5 + 0.5 * torch.sin(phase))   # [B,T,H,W]
        a = mask * amp
        out = linear_to_srgb((1 - a) * srgb_to_linear(clean) + a * stripe[:, None])
        return clean + mask * (out - clean), amp, par


STRIPE_REV_HZ = (2.0, 10.0)         # 줄무늬 v2 명암 반전 주파수 범위 (24 fps 의 나이퀴스트 12 Hz 아래)
PATTERN_SHAPES = ("straight", "concentric", "radial", "spiral", "chirp", "checker", "wave")
PATTERN_KO = {"straight": "직선", "concentric": "동심원", "radial": "방사형", "spiral": "나선", "chirp": "처프(원근)",
              "checker": "체커", "wave": "물결"}


def pattern_profile(p: torch.Tensor, h: int, w: int, t: int, fps: float, ppd: float) -> torch.Tensor:
    """무늬 하나의 명암 프로파일 [T,H,W] ∈ [−1, 1] (반전 전). p = par 한 줄 [13]:
    (cpd, θ, 속도 °/s, hi, 반전 Hz, 줄 모양 g, 반전 파형 g, 움직임, 떼어 놓음, 종류, 중심 x, 중심 y, 추가 값).

    stripe_freq.pattern_phase 와 같은 식(줄 = sin 위상)에 움직임을 더했다. 흐름은 종류마다:
      직선·체커·처프·물결 = 줄에 수직으로 이동, 동심원 = 고리가 퍼지거나 모임, 방사형·나선 = 회전
      (회전 각속도는 기준 반지름 r0 = 화면 높이의 1/4 에서 선속도가 '속도 °/s' 가 되게 — 픽셀 깜빡임 Hz = 속도 × cpd 로 같음)
    방사형 살 수 = r0 에서 국소 cpd 가 cpd 가 되게(중심에 가까울수록 높고 멀수록 낮음 — 판정 대역 밖이 생길 수 있음).
    추가 값: 처프 = 주파수가 화면 끝에서 1 + 3·값 배, 물결 = 진폭 0.2~1.2 주기·파장 2~6 주기, 나선 = 감김 방향."""
    dev = p.device
    cpd, theta, speed, kind = p[0], p[1], p[2], int(p[9])
    shape = PATTERN_SHAPES[kind]
    f = cpd / ppd                                                          # cycles/px
    yy = torch.arange(h, device=dev, dtype=torch.float32)[:, None].expand(h, w)
    xx = torch.arange(w, device=dev, dtype=torch.float32)[None, :].expand(h, w)
    dy, dx = yy - p[11] * h, xx - p[10] * w
    r = torch.sqrt(dx * dx + dy * dy) + 1e-6
    ang = torch.atan2(dy, dx)
    u = dx * torch.cos(theta) + dy * torch.sin(theta)
    v = -dx * torch.sin(theta) + dy * torch.cos(theta)
    tt = torch.arange(t, device=dev, dtype=torch.float32)[:, None, None] / fps
    shift = speed * ppd * tt                                               # px
    r0 = h / 4.0
    omega = speed * ppd / r0                                               # rad/s
    if shape in ("straight", "checker"):
        phi = 2 * math.pi * f * (u[None] - shift)
    elif shape == "concentric":
        phi = 2 * math.pi * f * (r[None] - shift)
    elif shape == "radial":
        spokes = torch.clamp(torch.round(2 * math.pi * r0 * f), min=4)
        phi = spokes * (ang[None] - omega * tt)
    elif shape == "spiral":
        spokes = torch.clamp(torch.round(2 * math.pi * r0 * f / 2), min=2) * (1 if float(p[12]) < 0.5 else -1)
        phi = 2 * math.pi * f * r[None] + spokes * (ang[None] - omega * tt)
    elif shape == "chirp":
        span = float(max(h, w))
        u0 = u - u.min()
        k = 1 + 3 * p[12]
        phi = 2 * math.pi * f * (u0[None] + (k - 1) * u0[None] ** 2 / (2 * span) - shift)
    else:                                                                  # wave
        amp = (0.2 + 1.0 * p[12]) / f
        lam = (2 + 4 * p[12]) / f
        phi = 2 * math.pi * f * (u[None] + amp * torch.sin(2 * math.pi * v[None] / lam) - shift)
    g = p[5]
    prof = torch.tanh(g * torch.sin(phi)) / torch.tanh(g)
    if shape == "checker":
        phi2 = 2 * math.pi * f * (v[None] - 0 * shift)
        prof = prof * torch.tanh(g * torch.sin(phi2)) / torch.tanh(g)
    return prof
DECOUPLE_HZ = 2.5                   # 줄무늬 v4: 떼어 놓은 클립의 픽셀 깜빡임 상한(Hz) — 휘도 기준 초당 3회 초과보다 아래


def stripe_reversal(par: torch.Tensor, t: int, fps: float) -> torch.Tensor:
    """줄무늬 v2 의 명암 반전 계수 [B,T] ∈ [−1, 1]. 반전이 아니면 1. 사각에 가까우면 반주기마다 부호만 바뀐다."""
    tt = torch.arange(t, device=par.device, dtype=torch.float32)[None] / fps
    rev_hz, g = par[:, 4:5], par[:, 6:7]
    c = torch.cos(2 * math.pi * rev_hz * tt)
    r = torch.tanh(g * c) / torch.tanh(g)
    return torch.where(rev_hz > 0, r, torch.ones_like(r))


def make_generator(hazard: str, width: int = 16, zdim: int = 8, fps: float = 24.0, red_palette: bool | str = False,
                   lum_smooth: bool = False, stripe_cond: str = "mlp") -> LightGenerator:
    """종류 이름 → 생성기. 체크포인트 config.hazard(없으면 'lum')·config.red_palette·config.lum_smooth(없으면 False)로 고른다."""
    if hazard not in HAZARDS:
        raise ValueError(f"hazard 는 {HAZARDS} 중 하나: {hazard}")
    if hazard == "lum" and lum_smooth:
        g = LightGenerator(width, zdim, fps)
        g.smooth = True
        return g
    if hazard == "red":
        return RedGenerator(width, zdim, fps, palette=red_palette)
    if hazard == "stripe":
        if stripe_cond in ("z2", "z2dc", "z3"):   # v3: v2 + 덮는 정도의 시간 흐림(z 마지막 성분, σ 0~12 프레임). zdim ≥ 9
            # z3(v5): z2dc + 무늬 종류 7가지(PATTERN_SHAPES). zdim ≥ 14
            # z2dc(v4): v3 와 같은 가중치에 '휘도 위험 떼어 놓기'만 더한다(G 는 덮는 정도만 정하므로 다시 학습할 필요 없음)
            # v2 의 '정지' 줄무늬는 덮는 정도 a 가 프레임마다 켜졌다 꺼졌다 했다(픽셀별 시간 표준편차 0.38~0.46) — 무늬는
            # 정지인데 깜빡이는 줄무늬였다. 그 데이터로 학습한 탐지기는 규칙 생성기의 진짜 정지 줄무늬를 7개 중 0개 찾았다
            # (checks/gdr/v6_roi_c_val_rule.md, 반전 7/8·흐름 8/9). 흐림이 크면 a 가 거의 일정해진다.
            if zdim < 9:
                raise ValueError("stripe_cond='z2' 는 zdim ≥ 9 가 필요합니다")
            g = StripeGenerator(width, zdim, fps, cond="z")
            g.smooth, g.smooth_max, g.smooth_radius = True, 12.0, 18
            g.decouple = stripe_cond in ("z2dc", "z3")
            g.patterns = stripe_cond == "z3"
            if g.patterns and zdim < 14:
                raise ValueError("stripe_cond='z3' 는 zdim ≥ 14 가 필요합니다")
            return g
        return StripeGenerator(width, zdim, fps, cond=stripe_cond)
    return {"lum": LightGenerator, "stripe": StripeGenerator}[hazard](width, zdim, fps)


def soft_judge_stripe(amp_m: torch.Tensor, hi: torch.Tensor, fps: float, *, tau_c: float = 0.03, tau_a: float = 0.03) -> dict:
    """줄무늬 대리 판정. amp_m = M·a [B,1,T,H,W], hi [B] → 줄무늬 면적 [B,T], 후보 [B,T], 1초 창 안 최대 지속(초) [B].
    hi 가 [B,T] 이면 프레임마다 명암이 다르다(v2 의 반전: 반전이 0 을 지나는 프레임은 명암이 거의 없다)."""
    hi_e = hi[:, None, None, None] if hi.dim() == 1 else hi[:, :, None, None]
    contrast = amp_m[:, 0] * (hi_e - STRIPE_LO)
    present = torch.sigmoid((contrast - STRIPE_CONTRAST_MIN) / tau_c)
    area = present.mean(dim=(2, 3))
    cand = torch.sigmoid((area - measure.PATTERN_AREA_STATIC) / tau_a)
    w = measure.window_frames(fps)
    kernel = torch.ones(1, 1, w, device=cand.device, dtype=cand.dtype)
    dur = F.conv1d(cand.unsqueeze(1), kernel, padding=w - 1).squeeze(1).amax(dim=1) / fps
    return {"duration": dur, "area": area, "candidate": cand}


class FrameDiscriminator(nn.Module):
    """프레임 한 장 [N,3,H,W] → 패치별 '진짜 같음' 점수. 깜빡임(시간 변화)은 보지 않는다."""

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GroupNorm(min(8, 2 * w), 2 * w), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.GroupNorm(min(8, 4 * w), 4 * w), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * w, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================================
# 4. 손실 함수
# ============================================================================================
HAZARD_MIN = 8.0          # 위험 목표: 소프트 전이 수 ≥ 8 (하드 기준 7번 이상에 여유 1)
NEAR_RANGE = (2.0, 5.0)   # 문턱 근처 안전 목표: 전이 2~5번 (하드 기준 6번 이하에 여유 1)


def count_loss(cnt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1초 전이 수를 목표에 맞추는 힌지 손실. 위험: count ≥ 8, 근처 안전: 2 ≤ count ≤ 5 (휘도·적색 공통)."""
    hazard = F.relu(HAZARD_MIN - cnt)
    near = F.relu(cnt - NEAR_RANGE[1]) + F.relu(NEAR_RANGE[0] - cnt)
    return torch.where(target > 0.5, hazard, near).mean()


def judge_loss(y: torch.Tensor, target: torch.Tensor, fps: float):
    """목표 판정에 맞추는 손실(힌지). 위험: count ≥ 8 이 되도록, 근처 안전: 2 ≤ count ≤ 5 가 되도록."""
    cnt = soft_judge(y, fps)["count"]
    return count_loss(cnt, target), cnt.detach()


def judge_loss_red(lin: torch.Tensor, target: torch.Tensor, fps: float):
    """적색: 선형 BGR [B,3,T,H,W] 의 1초 적색 전이 수를 목표에 맞춘다(기준 횟수는 휘도와 같다)."""
    cnt = soft_judge_red(lin, fps)["count"]
    return count_loss(cnt, target), cnt.detach()


def red_crisp_loss(rho: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """적색 ρ 를 '한 프레임에 끝나는 켜짐·꺼짐'으로 모는 손실 (2026-11-14, train_gan --lambda-crisp).

    하드 판정(measure.red_transitions)은 색이 바뀐 프레임을 순서대로 둘씩 짝지어, 두 변화가 모두 조건을 넘고 방향이
    반대인 쌍만 센다. ρ 가 여러 프레임에 걸쳐 서서히 바뀌면 작은 변화들이 짝에 끼어 쌍이 깨진다. 소프트 판정기는 이
    짝짓기를 흉내 내지 않아 그런 영상을 과대 계수했다(첫 학습 r_div2.0_s0: 위험 목표 48개 중 6개가 소프트 8~13 인데
    하드는 3~6 전이). 그래서 ρ 가 0·1 에 붙고(ρ(1−ρ)), 프레임 사이 변화가 0 또는 1 이 되도록(min(|Δρ|, 1−|Δρ|)) 한다.
    마스크 안에서만 잰다."""
    m = mask.expand_as(rho)
    n = m.sum().clamp(min=1.0)
    binar = (rho * (1 - rho) * m).sum() / n
    d = (rho[:, :, 1:] - rho[:, :, :-1]).abs()
    step = (torch.minimum(d, 1 - d) * m[:, :, 1:]).sum() / m[:, :, 1:].sum().clamp(min=1.0)
    return binar + step


def judge_loss_stripe(amp_m: torch.Tensor, hi: torch.Tensor, target: torch.Tensor, fps: float):
    """줄무늬: 위험 = 면적 > 40% 인 프레임이 1초 창 안에 0.7초 이상, 근처 안전 = 0.25초 이하이면서 줄무늬는 보임(면적 ≥ 20%)."""
    r = soft_judge_stripe(amp_m, hi, fps)
    hazard = F.relu(STRIPE_HAZARD_SEC - r["duration"])
    near = F.relu(r["duration"] - STRIPE_NEAR_SEC) + F.relu(STRIPE_NEAR_AREA - r["area"].amax(dim=1))
    return STRIPE_LOSS_SCALE * torch.where(target > 0.5, hazard, near).mean(), r["duration"].detach()


def stripe_diversity(par1: torch.Tensor, par2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """줄무늬 다양성: 같은 조건에서 z 가 다르면 줄무늬 조건(cpd·방향·속도·밝기)이 달라지도록(음수로 최소화).

    네 조건의 차이를 각각 0~1 로 맞춰 평균한다: 주파수는 로그 비, 방향은 |sin Δθ|(180° 주기), 속도·밝기는 범위로 나눔."""
    d_cpd = (torch.log(par1[:, 0]) - torch.log(par2[:, 0])).abs() / math.log(STRIPE_CPD[1] / STRIPE_CPD[0])
    d_th = torch.sin(par1[:, 1] - par2[:, 1]).abs()
    d_sp = (par1[:, 2] - par2[:, 2]).abs() / (2 * STRIPE_SPEED_MAX)
    d_hi = (par1[:, 3] - par2[:, 3]).abs() / (STRIPE_HI[1] - STRIPE_HI[0])
    gap = (d_cpd + d_th + d_sp + d_hi) / 4
    return -(gap / ((z1 - z2).abs().mean(1) + 1e-6)).mean()


def spectral_diversity(delta1, delta2, mask, z1, z2, normalize: bool = False):
    """다양성(mode seeking): 같은 조건에서 z 가 다르면 '빛의 시간 리듬'이 달라지도록.

    마스크 안 평균 빛 s(t) 의 주파수 스펙트럼 |FFT(s)| 차이 / z 차이 를 키운다(음수로 최소화).
    픽셀 차이가 아니라 리듬 차이를 보므로, 진폭만 바꾸는 쉬운 길(E2b)보다 주파수를 바꾸게 유도한다.
    normalize=True (2026-11-14, 실험용): 스펙트럼을 각자의 크기(L2)로 나눈 뒤 비교한다. 정규화하지 않으면 스펙트럼
      크기가 빛의 세기에 비례해서, G 가 리듬 대신 세기를 바꿔도 손실이 줄어든다 — λ_div 를 키울수록 빛이 세지고
      질감이 줄던 원인으로 추정한 것(checks/gan/u_sweep_SUMMARY.md §C). 정규화하면 세기만 다른 두 빛은 차이 0 이다.
    """
    msum = mask.sum(dim=(3, 4))[:, 0].clamp_min(1e-6)               # [B,1] or [B,T]
    s1 = (delta1 * mask).sum(dim=(3, 4))[:, 0] / msum                # [B,T]
    s2 = (delta2 * mask).sum(dim=(3, 4))[:, 0] / msum
    f1 = torch.fft.rfft(s1 - s1.mean(1, keepdim=True), dim=1).abs()
    f2 = torch.fft.rfft(s2 - s2.mean(1, keepdim=True), dim=1).abs()
    if normalize:
        f1 = f1 / (f1.norm(dim=1, keepdim=True) + 1e-6)
        f2 = f2 / (f2.norm(dim=1, keepdim=True) + 1e-6)
    gap = (f1 - f2).abs().mean(1)
    return -(gap / ((z1 - z2).abs().mean(1) + 1e-6)).mean()


def moving_mask(mask: torch.Tensor, frames: int, fps: float, velocity: torch.Tensor) -> torch.Tensor:
    """Translate a static mask over time. Velocity is normalized frame widths/heights per second.

    Returns [B,1,T,H,W]; clipped at image boundaries. Velocity is specified per clip.
    """
    if mask.ndim != 5 or mask.shape[1:3] != (1, 1) or velocity.shape != (mask.shape[0], 2):
        raise ValueError("expected mask [B,1,1,H,W] and velocity [B,2]")
    if frames < 1 or fps <= 0 or not math.isfinite(fps):
        raise ValueError("frames and fps must be positive")
    b, _, _, h, w = mask.shape
    times = torch.arange(frames, device=mask.device, dtype=mask.dtype) / fps
    theta = torch.eye(2, 3, device=mask.device, dtype=mask.dtype)[None, None].repeat(b, frames, 1, 1)
    theta[:, :, :, 2] = -2 * velocity.to(device=mask.device, dtype=mask.dtype)[:, None, :] * times[None, :, None]
    grid = F.affine_grid(theta.reshape(-1, 2, 3), (b * frames, 1, h, w), align_corners=False)
    base = mask[:, :, 0].repeat_interleave(frames, dim=0)
    shifted = F.grid_sample(base, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return shifted.reshape(b, frames, 1, h, w).permute(0, 2, 1, 3, 4)


def random_frames(video: torch.Tensor, k: int) -> torch.Tensor:
    """[B,3,T,H,W] 에서 클립마다 무작위 k 프레임 → [B·k,3,H,W] (D 입력)."""
    b, _, t, h, w = video.shape
    idx = torch.randint(0, t, (b, k), device=video.device)
    frames = torch.stack([video[i, :, idx[i]] for i in range(b)])   # [B,3,k,H,W]
    return frames.transpose(1, 2).reshape(b * k, 3, h, w)
