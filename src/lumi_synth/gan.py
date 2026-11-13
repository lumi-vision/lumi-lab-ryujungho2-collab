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
    a_up, a_dn = up.mean(dim=(2, 3)), dn.mean(dim=(2, 3))  # 방향별 면적 [B,T]
    e_up = torch.sigmoid((a_up - measure.AREA_LIMIT) / tau_a)
    e_dn = torch.sigmoid((a_dn - measure.AREA_LIMIT) / tau_a)
    new_up = e_up * (1 - F.pad(e_up[:, :-1], (1, 0)))
    new_dn = e_dn * (1 - F.pad(e_dn[:, :-1], (1, 0)))
    onset = torch.maximum(new_up, new_dn)                # D4: 같은 프레임 두 방향 = 1번
    w = measure.window_frames(fps)                       # 판정기와 같은 1초 창 (ceil)
    kernel = torch.ones(1, 1, w, device=y.device, dtype=onset.dtype)
    counts = F.conv1d(onset.unsqueeze(1), kernel, padding=w - 1).squeeze(1)
    soft_max = counts.amax(dim=1)
    return {"count": soft_max, "onset": onset, "area_up": a_up, "area_down": a_dn}


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
        return F.interpolate(delta, size=(t, h, w), mode="trilinear", align_corners=False)

    def forward(self, clean, mask, target, z):
        delta = self.light(clean, mask, target, z)
        lin = srgb_to_linear(clean) + mask * delta          # 세 채널에 같은 빛 → 휘도가 정확히 Δ 만큼 오름
        out = linear_to_srgb(lin)
        return clean + mask * (out - clean), delta          # 마스크 밖은 정확히 원본


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


def judge_loss(y: torch.Tensor, target: torch.Tensor, fps: float):
    """목표 판정에 맞추는 손실(힌지). 위험: count ≥ 8 이 되도록, 근처 안전: 2 ≤ count ≤ 5 가 되도록."""
    cnt = soft_judge(y, fps)["count"]
    hazard = F.relu(HAZARD_MIN - cnt)
    near = F.relu(cnt - NEAR_RANGE[1]) + F.relu(NEAR_RANGE[0] - cnt)
    return torch.where(target > 0.5, hazard, near).mean(), cnt.detach()


def spectral_diversity(delta1, delta2, mask, z1, z2):
    """다양성(mode seeking): 같은 조건에서 z 가 다르면 '빛의 시간 리듬'이 달라지도록.

    마스크 안 평균 빛 s(t) 의 주파수 스펙트럼 |FFT(s)| 차이 / z 차이 를 키운다(음수로 최소화).
    픽셀 차이가 아니라 리듬 차이를 보므로, 진폭만 바꾸는 쉬운 길(E2b)보다 주파수를 바꾸게 유도한다.
    """
    msum = mask.sum(dim=(2, 3, 4)) + 1e-6                            # [B,1]
    s1 = (delta1 * mask).sum(dim=(3, 4))[:, 0] / msum                # [B,T]
    s2 = (delta2 * mask).sum(dim=(3, 4))[:, 0] / msum
    f1 = torch.fft.rfft(s1 - s1.mean(1, keepdim=True), dim=1).abs()
    f2 = torch.fft.rfft(s2 - s2.mean(1, keepdim=True), dim=1).abs()
    gap = (f1 - f2).abs().mean(1)
    return -(gap / ((z1 - z2).abs().mean(1) + 1e-6)).mean()


def random_frames(video: torch.Tensor, k: int) -> torch.Tensor:
    """[B,3,T,H,W] 에서 클립마다 무작위 k 프레임 → [B·k,3,H,W] (D 입력)."""
    b, _, t, h, w = video.shape
    idx = torch.randint(0, t, (b, k), device=video.device)
    frames = torch.stack([video[i, :, idx[i]] for i in range(b)])   # [B,3,k,H,W]
    return frames.transpose(1, 2).reshape(b * k, 3, h, w)
