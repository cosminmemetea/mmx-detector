import uuid

from pydantic import BaseModel, Field


class DetectionTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class DetectionResponse(BaseModel):
    task_id: str
    detections: list[dict]
