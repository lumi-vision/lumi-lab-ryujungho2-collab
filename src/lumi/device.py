"""학습·생성 스크립트가 공유하는 연산 장치 선택과 CUDA 진단."""
from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """`auto`는 CUDA가 있으면 CUDA, 없으면 CPU. `cuda`를 요구했는데 없으면 즉시 실패한다."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA를 요청했지만 사용할 수 없습니다. `python scripts/check_env.py`로 "
            "PyTorch CUDA 빌드와 드라이버를 확인하세요 (docs/HOME_GPU_SETUP.md)."
        )
    return torch.device(name)


def configure_cuda(device: torch.device) -> None:
    """Ampere 이후 GPU에서 TF32 행렬곱과 cuDNN 자동 튜닝을 켠다. CPU에서는 아무것도 하지 않는다."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def arch_supported(index: int = 0) -> bool:
    """설치된 PyTorch 빌드가 해당 GPU의 compute capability용 커널을 포함하는지 확인한다.

    RTX 50 시리즈(sm_120)는 CUDA 12.8 이상으로 빌드된 PyTorch가 필요하다.
    """
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


def describe(device: torch.device) -> str:
    if device.type != "cuda":
        return f"{device} (torch {torch.__version__})"
    index = device.index or 0
    props = torch.cuda.get_device_properties(index)
    return (f"{device} · {props.name} · {props.total_memory / 2**30:.1f} GiB · "
            f"sm_{props.major}{props.minor} · torch {torch.__version__} / CUDA {torch.version.cuda}")
