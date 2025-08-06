import json
import redis.asyncio as redis
from typing import Annotated, Optional
import time
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import os

from celeb_video_annotator.utils import load_config
from celeb_video_annotator.api.schema import *

app = FastAPI()

# Initialize async Redis connection with error handling
redis_client = None

async def get_redis():
    """Get Redis connection, initialize if needed"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await redis_client.ping()
            print("✅ Redis connected successfully")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            redis_client = None
    return redis_client

async def store_job(job_id: str, job_data: dict):
    """Store job in Redis (async)"""
    redis = await get_redis()
    if redis is None:
        print("❌ Redis not available, cannot store job")
        return
    await redis.set(f"job:{job_id}", json.dumps(job_data, default=str))

async def get_job(job_id: str) -> dict:
    """Get job from Redis (async)"""
    redis = await get_redis()
    if redis is None:
        print("❌ Redis not available, cannot get job")
        return None
    job_json = await redis.get(f"job:{job_id}")
    if not job_json:
        return None
    return json.loads(job_json)

async def update_job_status(job_id: str, status: str, progress: float = None, download_url: str = None, errors: list = None):
    """Update job status in Redis (async)"""
    job_data = await get_job(job_id)
    if job_data:
        job_data["status"] = status
        if progress is not None:
            job_data["progress"] = progress
        if download_url is not None:
            job_data["download_url"] = download_url
        if errors is not None:
            job_data["errors"] = errors
        await store_job(job_id, job_data)

async def annotate_video_background(job_id: str, video_path: str, config: dict, recognizer = None, save_results=False):
    try:

        # 1. Initialize processing
        await update_job_status(job_id, status="running", progress=5.0)

        # 2. Load face recognizer
        if recognizer is None:
            from celeb_video_annotator.core.face_recognizer import AutomaticFaceRecognizer
            recognizer = AutomaticFaceRecognizer(config)
        await update_job_status(job_id, status="running", progress=20.0)

        # 3. Extract faces from video
        results = recognizer.extract_and_embeddings_from_video(video_path)
        await update_job_status(job_id, status="running", progress=60.0)

        # 4. Save results if necessary
        if save_results:
            base_name = Path(video_path).stem
            results_path = f"{config['output_dir']}/{base_name}_recognition.json"
            recognizer.save_results(results, results_path)
        
        # 5. Create annotated video
        annotated_video_path = f"results/{job_id}_annotated.mp4"
        recognizer.rebuild_video_with_annotations(
            video_path,
            results,
            annotated_video_path,
            font_scale=0.8,
            box_color=(0, 255, 0),  # Green boxes
            text_color=(255, 255, 255),  # White text
            box_thickness=3,
            min_confidence=0.8,
            only_label=config['target_label'] 
        )
        await update_job_status(job_id, status="running", progress=95.0)

        # 6. Mark as completed
        download_url = f"/download/{job_id}"
        await update_job_status(job_id, status="completed", progress=100.0, download_url=download_url)

        # 7. Cleanup temp file
        if os.path.exists(video_path):
            os.remove(video_path)

    except Exception as e:
        # Update with error
        await update_job_status(job_id, status="failed", errors=[str(e)])

@app.post("/annotate")
async def annotate_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to annotate"),
    only_label: Optional[str] = None
):
    # use pre-set recognizer
    recognizer = None

    # create job ID
    job_id = str(uuid.uuid4())

    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    # Save uploaded file to temp location
    temp_file_path = temp_dir / f"{job_id}_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # create job data to store on Redis
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "download_url": "none",
        "errors": [],
    }
    await store_job(job_id, job_data)

    # load config
    config = load_config('config/config.yaml')
    
    if only_label is not None:
        config['target_label'] = only_label

    # Ensure results path exists
    Path("results").mkdir(exist_ok=True)

    # Create background task to annotate video
    background_tasks.add_task(
        annotate_video_background,
        job_id,
        str(temp_file_path),
        config
    )

    return AnnotationJobResponse(
        job_id=job_id,
        status="pending",
        filename=file.filename,
        file_size=file.size,
        submitted_at=time.time()
    )

@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    job_data = await get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")
    return AnnotationStatusResponse(**job_data)

@app.get("/jobs/{job_id}/download")
async def download_video(job_id: str):
    job_data = await get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")
    
    if job_data["status"] != "completed":
        raise HTTPException(400, "Video not ready")
    
    # Find annotated video file
    video_files = list(Path("results").glob(f"{job_id}*_annotated.mp4"))
    if not video_files:
        raise HTTPException(404, "Video file not found")
    
    return FileResponse(
        path=str(video_files[0]),
        media_type="video/mp4",
        filename=f"annotated_{job_data.get('filename', 'video')}"
    )

@app.get("/health")
async def health_check():
    status = {"status": "ok", "checks": {}}
    # Test Redis
    try:
        redis = await get_redis()
        if redis is None:
            status["checks"]["redis"] = "❌ Not initialized"
        else:
            await redis.ping()
            status["checks"]["redis"] = "✅ Connected"
    except Exception as e:
        status["checks"]["redis"] = f"❌ Failed: {e}"
    
    # Test config loading
    try:
        config = load_config('config/config.yaml')
        status["checks"]["config"] = "✅ Loaded"
    except Exception as e:
        status["checks"]["config"] = f"❌ Failed: {e}"
    
    # Test face recognizer import
    try:
        from celeb_video_annotator.core.face_recognizer import AutomaticFaceRecognizer
        status["checks"]["face_recognizer"] = "✅ Can import"
    except Exception as e:
        status["checks"]["face_recognizer"] = f"❌ Failed: {e}"
    
    return status

    

    
