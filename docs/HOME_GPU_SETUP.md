# 노트북 ↔ GitHub ↔ 집 GPU PC 작업 흐름

| 장비 | 역할 | 사양 |
|---|---|---|
| 노트북 | 코드·문서 수정, CPU 단위 테스트 | CUDA 없음 |
| GitHub `lumi-vision/lumi-lab-ryujungho2-collab` | 코드·문서 동기화 (데이터·가중치 제외) | — |
| 집 PC | 데이터 생성, GAN 학습, 미리보기 | RTX 5080 16GB · RAM 64GB · SSD 2TB |

데이터(`datasets/`), 체크포인트(`checkpoints/`), 영상(`sample/`, `*.avi`, `*.mp4`)은 `.gitignore`로 제외한다. 집 PC에서 같은 시드로 **다시 생성**하고, 옮겨야 하면 외장 SSD 등 git 밖의 경로를 쓴다.

## 1. 집 PC 최초 설치 (1회)

준비물: 최신 NVIDIA 드라이버(Game Ready/Studio), Python 3.10 이상, Git.

```powershell
git clone https://github.com/lumi-vision/lumi-lab-ryujungho2-collab.git
cd lumi-lab-ryujungho2-collab
Set-ExecutionPolicy -Scope Process Bypass
.\setup_gpu_windows.ps1            # torch CUDA 13.2 빌드(cu132)
# 드라이버가 CUDA 13.2를 지원하지 않는다는 오류가 나면:
# .\setup_gpu_windows.ps1 -CudaTag cu130
```

**RTX 5080 주의:** Blackwell(sm_120) 커널은 **CUDA 12.8 이상 빌드의 PyTorch**에만 들어 있다. 기본 `pip install torch`는 Windows에서 CPU 빌드를 설치하므로 반드시 `--index-url`로 CUDA 빌드를 지정한다. 스크립트가 그렇게 설치한다. 확인 명령:

```powershell
.venv\Scripts\python.exe scripts\check_env.py --require-cuda
```

`[OK] 아키텍처 커널`과 `[OK] Conv3d + bf16 autocast`가 나와야 한다. `nvidia-smi` 오른쪽 위의 "CUDA Version"은 드라이버가 지원하는 **최대** 버전이며, 설치한 휠 태그(cu132 = 13.2)보다 크거나 같아야 한다.

## 2. 첫 실행

```powershell
.venv\Scripts\Activate.ps1
python scripts/generate_demo.py --output sample/lumi_demo.avi

# 미리보기: 기본 일시정지 상태로 열림. SPACE 재생 · N/P 프레임 이동 · Q 종료
# 반드시 직접 확인한 비위험 소스에만 --allow-preview를 쓴다 (합성 결과물에는 쓰지 않는다)
python -m lumi preview sample/lumi_demo.avi --allow-preview

python scripts/build_dataset.py --source sample/lumi_demo.avi --output datasets/oracle --samples 200 --frames 16 --width 128 --height 96 --fps 24 --seed 42
python scripts/audit_dataset.py --dataset datasets/oracle

# GPU 학습: CUDA에서는 bf16 autocast가 자동으로 켜진다 (--amp none으로 끔)
python scripts/train_gan.py --dataset datasets/oracle --device cuda --epochs 20 --batch-size 8 --workers 4 --width 32 --checkpoint checkpoints/lumi_cgan.pt
python scripts/generate_gan.py --dataset datasets/oracle --device cuda --checkpoint checkpoints/lumi_cgan.pt --output datasets/gan --per-clip 2
python scripts/audit_dataset.py --dataset datasets/gan
```

`--device cuda`를 주면 CUDA가 없을 때 CPU로 조용히 넘어가지 않고 바로 실패한다. 학습이 끝나면 `GPU 최대 메모리 사용`이 출력되니, 16GB 안에서 `--batch-size`와 `--width`를 조정한다. 위 배치/폭 값은 출발점일 뿐 측정된 최적값이 아니다.

## 3. 일상 동기화

```powershell
# 작업 시작 전 (어느 장비든)
git pull --rebase

# 작업 후
git add -A
git status          # .npz/.pt/.avi가 보이면 커밋하지 않는다
git commit -m "feat: ..."
git push
```

학습 결과 수치(손실 로그, 메모리, 소요 시간)는 `docs/` 아래 실험 기록에 텍스트로 남겨 커밋한다. 체크포인트 파일 자체는 올리지 않는다.
