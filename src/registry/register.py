import json

import mlflow
import mlflow.pytorch
import torch

from src.config import PROJECT_ROOT, load_config
from src.models.classifier import create_resnet18


def check_file_exists(path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def main() -> None:
    config = load_config()

    model_name = config["mlflow"]["registered_model_name"]
    checkpoint_path = PROJECT_ROOT / "artifacts" / "best_model.pt"
    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"

    check_file_exists(checkpoint_path, "model checkpoint")
    check_file_exists(metrics_path, "evaluation metrics")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    macro_f1 = metrics["macro_f1"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]

    model = create_resnet18(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="register_resnet18_classifier") as run:
        mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_param("classes", ",".join(classes))

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=model_name,
            input_example=torch.randn(1, 3, 224, 224),
            serialization_format="pickle",
        )

    print(f"Registered model name: {model_name}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Metrics file: {metrics_path}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"MLflow run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")


if __name__ == "__main__":
    main()
