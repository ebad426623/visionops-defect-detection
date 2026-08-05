from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

CHECKPOINT_PATH = Path("artifacts/best_model.pt")
if not CHECKPOINT_PATH.exists():
    pytest.skip(
        "API tests require artifacts/best_model.pt",
        allow_module_level=True,
    )

from api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_alias"] == "champion"


def test_feedback_valid_label() -> None:
    response = client.post(
        "/feedback",
        json={
            "prediction_id": "test-id",
            "correct_label": "patches",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "feedback recorded"


def test_feedback_invalid_label() -> None:
    response = client.post(
        "/feedback",
        json={
            "prediction_id": "test-id",
            "correct_label": "not_a_real_class",
        },
    )

    assert response.status_code == 400


def test_predict_valid_image() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(
        image_buffer,
        format="JPEG",
    )
    image_buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_buffer, "image/jpeg")},
    )

    body = response.json()

    assert response.status_code == 200
    assert "prediction_id" in body
    assert 0 <= body["confidence"] <= 1
    assert "probabilities" in body
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.001
