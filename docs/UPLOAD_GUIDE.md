# 개인 실험 저장소에 업로드하는 방법

**대상:** https://github.com/lumi-vision/lumi-lab-ryujungho2-collab  
**범위:** 소스 코드, 테스트, 연구 문서만. 팀 저장소 `lumi-vision/lumi`는 변경하지 않는다.

## Windows PowerShell

Git이 설치되고 GitHub 로그인/조직 저장소 쓰기 권한이 준비된 상태에서 실행한다.

```powershell
git clone https://github.com/lumi-vision/lumi-lab-ryujungho2-collab.git
cd lumi-lab-ryujungho2-collab

# ZIP 파일 이름/위치는 실제 다운로드 경로에 맞춘다.
Expand-Archive -Path "$HOME\Downloads\lumi_lab_hamsungdo_research_v03.zip" -DestinationPath . -Force

# 기존 저장소의 .gitignore를 보존하면서 필요한 제외 규칙을 추가한다.
@'
datasets/
checkpoints/
output/
sample/
*.pt
*.pth
*.ckpt
'@ | Add-Content .gitignore

git add README.md pyproject.toml run_windows.bat src scripts tests docs .gitignore
git diff --cached --stat
git status
git commit -m "docs: document evidence-based LUMI data and GAN research prototype"
git push origin main
```

**주의:** `git diff --cached --stat`에서 대용량 `.avi`, `.mp4`, `.npz`, `.pt`가 보이면 `git commit`을 실행하지 않는다. 압축 파일에는 이러한 데이터/가중치가 포함되지 않는다.

### 업로드 후 확인

- README에서 `연구 설계서`, `Barbu GAN`, `데이터셋 규약`, `실험 계획`의 링크가 열리는지 확인한다.
- `python -m pip install -e '.[dev]'` 후 `python -m pytest -q`를 실행한다.
- 위험한 합성 결과를 영상으로 자동 재생하는 도구를 추가하지 않는다.

**권한 문제:** ChatGPT의 GitHub 앱 연결은 GitHub 사용자에게 Push가 표시되어도 조직 저장소의 실제 Contents 쓰기에서 403이 발생할 수 있다. 이 경우 로컬 `git push`는 별도의 인증 경로이므로 성공할 수 있다. 조직 정책/branch protection에 따라 PR이 필요할 수도 있다.
