"""LUMI Safety Engine FastAPI 백엔드 서버.

안전 및 무결성 정책:
    - 학습된 가중치(lumi7_best.pth)가 로드되지 않은 상태에서는 /api/analyze-and-process 요청을
      HTTP 503(Service Unavailable)으로 거부하여 미학습 무작위 출력 반환을 차단한다.
    - 업로드 파일의 확장자(.avi, .mov 등)와 무관하게 출력 파일 컨테이너를 .mp4로 고정하고,
      VideoWriter 및 출력 파일 생성 검증 실패 시 HTTP 500 에러를 반환한다.

실행 방법:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from infer import LUMIInferencer

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = Path(os.getenv("LUMI_WEIGHTS_PATH", str(BASE_DIR / "lumi7_best.pth")))

app = FastAPI(title="LUMI Safety Engine API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _init_inferencer() -> tuple[Optional[LUMIInferencer], Optional[str]]:
    try:
        engine = LUMIInferencer(weights_path=str(WEIGHTS_PATH), allow_untrained=False)
        return engine, None
    except Exception as exc:
        return None, str(exc)


inferencer, init_error = _init_inferencer()


@app.get("/health")
def health_check():
    if inferencer is None or not inferencer.weights_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "UNAVAILABLE",
                "weights_loaded": False,
                "weights_path": str(WEIGHTS_PATH),
                "detail": init_error or "학습된 모델 가중치가 로드되지 않았습니다.",
            },
        )
    return {
        "status": "LUMI Engine Operational",
        "device": str(inferencer.device),
        "weights_loaded": inferencer.weights_loaded,
        "weights_path": str(WEIGHTS_PATH),
    }


@app.post("/api/analyze-and-process")
async def analyze_and_process_video(file: UploadFile = File(...)):
    global inferencer, init_error

    # 서버 기동 후 가중치 파일이 뒤늦게 배치된 경우 1회 재시도
    if inferencer is None or not inferencer.weights_loaded:
        inferencer, init_error = _init_inferencer()

    if inferencer is None or not inferencer.weights_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"모델 가중치가 준비되지 않아 추론을 수행할 수 없습니다: {init_error}",
        )

    job_id = uuid.uuid4().hex[:8]
    src_ext = Path(file.filename or "upload.mp4").suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}{src_ext}"

    # 출력 컨테이너는 항상 .mp4로 고정
    safe_out_name = f"safe_{job_id}.mp4"
    mask_out_name = f"mask_{job_id}.mp4"
    safe_out_path = OUTPUT_DIR / safe_out_name
    mask_out_path = OUTPUT_DIR / mask_out_name

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        report = inferencer.process_video(
            input_video_path=str(input_path),
            output_safe_path=str(safe_out_path),
            output_mask_path=str(mask_out_path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"영상 처리 중 오류가 발생했습니다: {exc}") from exc

    return JSONResponse(
        content={
            "status": "COMPLETED",
            "metadata": {
                "filename": file.filename,
                "total_frames": report["total_frames"],
                "source_fps": report["source_fps"],
                "fps": report["fps"],
            },
            "metrics": {
                "psnr_db": report["avg_psnr_db"],
                "processing_fps": report["processing_fps"],
                "detected_frames": report["detected_event_count"],
                "weights_loaded": report["weights_loaded"],
            },
            "events": report["sample_events"],
            "result_video_url": f"http://localhost:8000/static/outputs/{safe_out_name}",
            "mask_video_url": f"http://localhost:8000/static/outputs/{mask_out_name}",
        }
    )
