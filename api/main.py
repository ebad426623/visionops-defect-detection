import json
from datetime import datetime, timezone
from io import BytesIO
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import get_eval_transforms
from src.models.classifier import create_resnet18

app = FastAPI(title="VisionOps Defect Detection API")
config = load_config()
image_size = config["data"].get("image_size", 224)
transform = get_eval_transforms(image_size)
checkpoint_path = PROJECT_ROOT / "artifacts" / "best_model.pt"
prediction_log_path = PROJECT_ROOT / "artifacts" / "predictions.jsonl"
feedback_log_path = PROJECT_ROOT / "artifacts" / "feedback.jsonl"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model = create_resnet18(num_classes=len(checkpoint["classes"]), freeze_backbone=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prediction_requests = Counter(
    "visionops_prediction_requests_total",
    "Total prediction requests.",
)
prediction_errors = Counter(
    "visionops_prediction_errors_total",
    "Total prediction errors.",
)
prediction_latency = Histogram(
    "visionops_prediction_latency_seconds",
    "Prediction latency in seconds.",
)
prediction_confidence = Histogram(
    "visionops_prediction_confidence",
    "Prediction confidence.",
)
predicted_class_total = Counter(
    "visionops_predicted_class_total",
    "Predicted class counts.",
    ["class_name"],
)
active_model_info = Gauge(
    "visionops_active_model_info",
    "Active model metadata.",
    ["model_alias"],
)
active_model_info.labels(model_alias="champion").set(1)


class FeedbackRequest(BaseModel):
    prediction_id: str
    correct_label: str


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {
        "model_name": config["mlflow"]["registered_model_name"],
        "model_alias": "champion",
        "checkpoint_path": str(checkpoint_path),
    }


def log_prediction(record: dict[str, object]) -> None:
    prediction_log_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, separators=(", ", ": ")) + "\n")


def log_feedback(record: dict[str, object]) -> None:
    feedback_log_path.parent.mkdir(parents=True, exist_ok=True)

    with feedback_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, separators=(", ", ": ")) + "\n")


def get_image_stats(image: Image.Image) -> dict[str, float | int]:
    grayscale_image = image.convert("L")
    pixels = np.array(grayscale_image)

    return {
        "width": image.width,
        "height": image.height,
        "mean_brightness": float(pixels.mean()),
        "pixel_std": float(pixels.std()),
    }


def count_jsonl_records(path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    start_time = perf_counter()
    prediction_id = str(uuid4())
    prediction_requests.inc()

    if file.content_type not in {"image/jpeg", "image/png"}:
        prediction_errors.inc()
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        prediction_errors.inc()
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as error:
        prediction_errors.inc()
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable image.",
        ) from error

    image_stats = get_image_stats(image)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    prediction = checkpoint["classes"][predicted_index.item()]
    probability_map = {
        class_name: probabilities[index].item()
        for index, class_name in enumerate(checkpoint["classes"])
    }
    latency_ms = (perf_counter() - start_time) * 1000
    prediction_latency.observe(latency_ms / 1000)
    prediction_confidence.observe(confidence.item())
    predicted_class_total.labels(class_name=prediction).inc()

    response = {
        "prediction_id": prediction_id,
        "prediction": prediction,
        "confidence": confidence.item(),
        "probabilities": probability_map,
        "model_alias": "champion",
        "latency_ms": latency_ms,
    }

    log_prediction(
        {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence.item(),
            "model_alias": "champion",
            "latency_ms": latency_ms,
            "image_stats": image_stats,
        }
    )

    return response


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    if request.correct_label not in checkpoint["classes"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown label: {request.correct_label}",
        )

    log_feedback(
        {
            "prediction_id": request.prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correct_label": request.correct_label,
        }
    )

    return {"status": "feedback recorded"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


@app.get("/stats")
def stats() -> dict[str, int]:
    return {
        "prediction_log_records": count_jsonl_records(prediction_log_path),
        "feedback_log_records": count_jsonl_records(feedback_log_path),
    }
