from schema import *

import json
import redis
from typing import Annotated, Optional
import time
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import os

from ..utils import load_config

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, db=0)

def store_job(job_id: str, job_data: dict):
    # Store job in Redis
    r.set(f"job:{job_id}", json.dumps(job_data, default=str))

def get_job(job_id: str) -> dict:
    # Get job from Redis
    job_json = r.get(f"job:{job_id}")
    if not job_json:
        return None
    return json.loads(job_json)

def update_job_status(job_id: str, status: str, progress: float = None, download_url: str = None, errors: list = None):
    job_data = get_job(job_id)
    if job_data:
        job_data["status"] = status
        if progress is not None:
            job_data["progress"] = progress
        if download_url is not None:
            job_data["download_url"] = download_url
        if errors is not None:
            job_data["errors"] = errors
        store_job(job_id, job_data)

async def annotate_video_background(job_id: str, video_path: str, config: dict, recognizer = None, save_results=False):
    try:

        # 1. Initialize processing
        update_job_status(job_id, status="running", progress=5.0)

        # 2. Load face recognizer
        if recognizer is None:
            from ..core.face_recognizer import AutomaticFaceRecognizer
            recognizer = AutomaticFaceRecognizer(config)
        update_job_status(job_id, status="running", progress=20.0)

        # 3. Extract faces from video
        results = recognizer.extract_and_embeddings_from_video(video_path)
        update_job_status(job_id, status="running", progress=60.0)

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
        update_job_status(job_id, status="running", progress=95.0)

        # 6. Mark as completed
        download_url = f"/download/{job_id}"
        update_job_status(job_id, status="completed", progress=100.0, download_url=download_url)

        # 7. Cleanup temp file
        if os.path.exists(video_path):
            os.remove(video_path)

    except Exception as e:
        # Update with error
        update_job_status(job_id, status="failed", errors=[str(e)])

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
    store_job(job_id, job_data)

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
    job_data = get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")
    return AnnotationStatusResponse(**job_data)

@app.get("/jobs/{job_id}/download")
async def download_video(job_id: str):
    job_data = get_job(job_id)
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

@app.post("/x")
async def test(x: Annotated[int, Query(description="hi")]):
    return "hi"

    

    
