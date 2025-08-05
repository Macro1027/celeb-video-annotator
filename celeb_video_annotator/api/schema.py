from typing import Union, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import time


class AnnotationJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    filename: str
    file_size: int
    submitted_at: datetime

class AnnotationStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: Annotated[float, Field(ge=0.0, le=100.0)]
    download_url: str
    errors: list[str] = []

class AnnotationResult(BaseModel):
    pass

class Celebrities(BaseModel):
    name: str
    id: int
    thumbnail_url: Optional[str] = None

class CelebritiesList(BaseModel):
    celebrities: list[Celebrities]

class FaceDetection(BaseModel):
    frame_idx: int
    bounding_box: dict[str, float]
    celebrity_name: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    similarity_score: Optional[float] = None
    vote_count: Optional[int] = None
    confidence_ratio: Optional[float] = None