"""LUMI Safety Engine FastAPI 백엔드 서버.

안전 및 운영 품질 정책:
    1. 실제 데이터셋으로 검증된(validated=True, trained_on_dummy=False) 체크포인트가 로드되지
       않은 상태에서는 /api/analyze-and-process 요청을 HTTP 503으로 차단한다.
    2. 결과 비디오 URL은 하드코딩된 localhost 대신 요청 Host 기반 URL(request.url_for)과
       상대 경로(/static/outputs/...)를 함께 반환하여 원격·배포 환경에서도 동작한다.
    3. 업로드 파일 저장 및 프레임 단위 PyTorch/OpenCV 추론은 워커 스레드풀(run_in_threadpool)에서
       실행하여 긴 영상 처리 중에도 FastAPI 이벤트 루프와 /health 응답이 막히지 않는다.

실행 방법:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from infer import LUMIInferencer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = Path(os.getenv("LUMI_WEIGHTS_PATH", str(BASE_DIR / "lumi7_best.pth")))

app = FastAPI(title="LUMI Safety Engine API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="static")

_engine_lock = threading.Lock()


def _init_inferencer() -> tuple[Optional[LUMIInferencer], Optional[str]]:
    try:
        engine = LUMIInferencer(
            weights_path=str(WEIGHTS_PATH),
            allow_untrained=False,
            allow_unvalidated=False,
        )
        return engine, None
    except Exception as exc:
        return None, str(exc)


inferencer, init_error = _init_inferencer()


def _ensure_inferencer() -> LUMIInferencer:
    global inferencer, init_error
    if inferencer is not None and inferencer.weights_loaded and inferencer.validated:
        return inferencer
    with _engine_lock:
        if inferencer is not None and inferencer.weights_loaded and inferencer.validated:
            return inferencer
        inferencer, init_error = _init_inferencer()
        if inferencer is None or not inferencer.weights_loaded or not inferencer.validated:
            raise HTTPException(
                status_code=503,
                detail=f"검증된 모델 체크포인트가 준비되지 않아 추론을 수행할 수 없습니다: {init_error}",
            )
        return inferencer


def _save_upload_file(src_file: BinaryIO, dst_path: Path) -> None:
    with open(dst_path, "wb") as buffer:
        shutil.copyfileobj(src_file, buffer)


@app.get("/health")
def health_check():
    if inferencer is None or not inferencer.weights_loaded or not inferencer.validated:
        return JSONResponse(
            status_code=503,
            content={
                "status": "UNAVAILABLE",
                "weights_loaded": False,
                "validated": False,
                "weights_path": str(WEIGHTS_PATH),
                "detail": init_error or "검증된 모델 체크포인트가 로드되지 않았습니다.",
            },
        )
    return {
        "status": "LUMI Engine Operational",
        "device": str(inferencer.device),
        "weights_loaded": inferencer.weights_loaded,
        "validated": inferencer.validated,
        "weights_path": str(WEIGHTS_PATH),
    }


@app.post("/api/analyze-and-process")
async def analyze_and_process_video(request: Request, file: UploadFile = File(...)):
    engine = await run_in_threadpool(_ensure_inferencer)

    job_id = uuid.uuid4().hex[:8]
    src_ext = Path(file.filename or "upload.mp4").suffix or ".mp4"

    # 출력 컨테이너는 항상 .mp4로 고정
    safe_out_name = f"safe_{job_id}.mp4"
    mask_out_name = f"mask_{job_id}.mp4"
    safe_out_path = OUTPUT_DIR / safe_out_name
    mask_out_path = OUTPUT_DIR / mask_out_name

    def process_upload():
        with tempfile.TemporaryDirectory(prefix="lumi-upload-") as folder:
            input_path = Path(folder) / f"input{src_ext}"
            _save_upload_file(file.file, input_path)
            return engine.process_video(str(input_path), str(safe_out_path), str(mask_out_path))

    try:
        report = await run_in_threadpool(process_upload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"영상 처리 중 오류가 발생했습니다: {exc}") from exc

    safe_rel_url = f"/static/outputs/{safe_out_name}"
    mask_rel_url = f"/static/outputs/{mask_out_name}"
    safe_abs_url = str(request.url_for("static", path=safe_out_name))
    mask_abs_url = str(request.url_for("static", path=mask_out_name))

    return JSONResponse(
        content={
            "status": "COMPLETED",
            "metadata": {
                "filename": file.filename,
                "analyzed_frames": report["analyzed_frames"],
                "total_frames": report["total_frames"],
                "source_fps": report["source_fps"],
                "fps": report["fps"],
            },
            "metrics": {
                "psnr_db": report["avg_psnr_db"],
                "processing_fps": report["processing_fps"],
                "detected_frames": report["detected_event_count"],
                "weights_loaded": report["weights_loaded"],
                "validated": report["validated"],
            },
            "events": report["sample_events"],
            "result_video_path": safe_rel_url,
            "mask_video_path": mask_rel_url,
            "result_video_url": safe_abs_url,
            "mask_video_url": mask_abs_url,
        }
    )
