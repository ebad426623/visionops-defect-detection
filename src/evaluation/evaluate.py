import json

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import get_eval_transforms
from src.models.classifier import create_resnet18


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_examples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy, all_labels, all_predictions


def main() -> None:
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    image_size = config["data"].get("image_size", 224)
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config["data"].get("num_workers", 0)
    checkpoint_path = PROJECT_ROOT / "artifacts" / "best_model.pt"

    test_dataset = datasets.ImageFolder(
        root=processed_dir / "test",
        transform=get_eval_transforms(image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_resnet18(
        num_classes=len(checkpoint["classes"]),
        freeze_backbone=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, labels, predictions = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    report_text = classification_report(
        labels,
        predictions,
        target_names=test_dataset.classes,
        digits=4,
    )
    report_dict = classification_report(
        labels,
        predictions,
        target_names=test_dataset.classes,
        output_dict=True,
    )

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "classification_report": report_dict,
    }
    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Classification report:")
    print(report_text)
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
