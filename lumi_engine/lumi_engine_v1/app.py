"""LUMI Safety Engine FastAPI 백엔드 서버.

Next.js(v0.dev) 대시보드와 연동하여 업로드된 영상을 분석·보정하고
결과 비디오 URL과 진단 메트릭 JSON을 반환한다.

실행 방법:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from infer import LUMIInferencer

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LUMI Safety Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 비디오 파일 서빙
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# 서버 기동 시 모델 1회 로드
inferencer = LUMIInferencer(weights_path=str(BASE_DIR / "lumi7_best.pth"))


@app.get("/health")
def health_check():
    return {
        "status": "LUMI Engine Operational",
        "device": str(inferencer.device),
        "weights_loaded": inferencer.weights_loaded,
    }


@app.post("/api/analyze-and-process")
async def analyze_and_process_video(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    safe_filename = f"{job_id}_{file.filename}"
    input_path = UPLOAD_DIR / safe_filename
    safe_out_path = OUTPUT_DIR / f"safe_{safe_filename}"
    mask_out_path = OUTPUT_DIR / f"mask_{safe_filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    report = inferencer.process_video(
        input_video_path=str(input_path),
        output_safe_path=str(safe_out_path),
        output_mask_path=str(mask_out_path),
    )

    return JSONResponse(
        content={
            "status": "COMPLETED",
            "metadata": {
                "filename": file.filename,
                "total_frames": report["total_frames"],
                "fps": report["fps"],
            },
            "metrics": {
                "psnr_db": report["avg_psnr_db"],
                "processing_fps": report["processing_fps"],
                "detected_frames": report["detected_event_count"],
                "weights_loaded": report["weights_loaded"],
            },
            "events": report["sample_events"],
            "result_video_url": f"http://localhost:8000/static/outputs/safe_{safe_filename}",
            "mask_video_url": f"http://localhost:8000/static/outputs/mask_{safe_filename}",
        }
    )
