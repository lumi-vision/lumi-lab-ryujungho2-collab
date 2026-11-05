"""집 GPU PC/노트북 환경 진단: PyTorch CUDA, GPU 아키텍처, OpenCV 미리보기(GUI), PyAV.

종료 코드: 0 = 모두 정상, 1 = 필수 항목 실패. `--require-cuda`를 주면 CUDA 부재도 실패로 처리한다.
"""
import argparse
import platform
import sys


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else '!!'}] {label}: {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="LUMI 실행 환경 진단")
    parser.add_argument("--require-cuda", action="store_true", help="CUDA가 없으면 실패로 처리 (집 GPU PC용)")
    args = parser.parse_args()
    ok = True
    print(f"Python {platform.python_version()} · {platform.platform()}")

    # OpenCV: 미리보기 창은 highgui가 포함된 opencv-python이 필요하다(opencv-python-headless 불가).
    try:
        import cv2
        info = cv2.getBuildInformation()
        gui_line = next((line.strip() for line in info.splitlines() if line.strip().startswith("GUI:")), "GUI: ?")
        has_gui = "NONE" not in gui_line.upper()
        ok &= check("OpenCV", has_gui, f"{cv2.__version__} · {gui_line}"
                    + ("" if has_gui else " → opencv-python-headless 제거 후 opencv-python 설치"))
    except ImportError:
        ok &= check("OpenCV", False, "미설치 → setup_windows.ps1 실행")

    try:
        import av
        check("PyAV(선택)", True, av.__version__)
    except ImportError:
        check("PyAV(선택)", True, "미설치 · 정밀 PTS가 필요하면 setup_windows.ps1 실행")

    try:
        import torch
    except ImportError:
        check("PyTorch", False, "미설치 → setup_windows.ps1 실행 (docs/COLLABORATION.md)")
        return 1
    check("PyTorch", True, f"{torch.__version__} · CUDA 빌드 {torch.version.cuda or '없음(CPU 전용)'}")

    if not torch.cuda.is_available():
        cuda_ok = check("CUDA", not args.require_cuda,
                        "사용 불가 · CPU로 실행됩니다" if not args.require_cuda else
                        "사용 불가 → CPU 빌드이거나 드라이버 문제 (nvidia-smi 확인)")
        return 0 if ok and cuda_ok else 1

    from lumi.device import arch_supported, describe
    import torch.nn.functional as F
    device = torch.device("cuda")
    check("GPU", True, describe(device))
    ok &= check("아키텍처 커널", arch_supported(),
                f"빌드 포함 {torch.cuda.get_arch_list()}"
                + ("" if arch_supported() else " → RTX 50 시리즈는 cu128 이상 빌드 필요"))
    if not ok:
        return 1
    try:
        x = torch.rand(2, 3, 16, 96, 128, device=device)
        w = torch.rand(8, 3, 3, 3, 3, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = F.conv3d(x, w, padding=1)
        torch.cuda.synchronize()
        check("Conv3d + bf16 autocast", bool(torch.isfinite(y).all()), f"출력 {tuple(y.shape)}")
    except RuntimeError as error:
        ok &= check("Conv3d + bf16 autocast", False, str(error).splitlines()[0])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
