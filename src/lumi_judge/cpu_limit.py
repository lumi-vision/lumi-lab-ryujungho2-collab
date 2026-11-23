"""CPU 사용 제한(사용자 요청 2026-11-23): 논리 코어 mask·우선순위 '보통 이하'·torch/cv2 스레드 수. 결과에는 영향 없음."""
from __future__ import annotations

import os


def limit_cpu(mask: int = 0x000F) -> None:
    mask = int(os.environ.get("LUMI_CPU_MASK", str(mask)), 0)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.GetCurrentProcess()
        k32.SetProcessAffinityMask(h, mask)
        k32.SetPriorityClass(h, 0x00004000)
    n = max(1, bin(mask).count("1"))
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:  # noqa: BLE001
        pass
