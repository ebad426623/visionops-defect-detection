from pathlib import Path

from src.config import PROJECT_ROOT


def check_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def main() -> None:
    prediction_log_path = PROJECT_ROOT / "artifacts" / "predictions.jsonl"
    feedback_log_path = PROJECT_ROOT / "artifacts" / "feedback.jsonl"
    checkpoint_path = PROJECT_ROOT / "artifacts" / "best_model.pt"

    check_path(prediction_log_path, "prediction log")
    check_path(feedback_log_path, "feedback log")
    check_path(checkpoint_path, "current champion checkpoint")

    print("Retraining readiness check passed.")
    print("Current project stores feedback metadata, not uploaded images.")
    print("For retraining, add corrected images under a labeled feedback dataset.")
    print("Then run: dvc repro")


if __name__ == "__main__":
    main()
