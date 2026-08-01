from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import PROJECT_ROOT, load_config
from src.data.dataset import get_eval_transforms, get_train_transforms
from src.models.classifier import (
    create_resnet18,
    count_trainable_parameters,
    unfreeze_layer4,
)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

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

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy


def run_training_phase(
    phase_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    train_classes: list[str],
    best_validation_accuracy: float,
) -> float:
    print(f"Starting phase: {phase_name}")

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"{phase_name} epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | "
            f"val_acc={validation_accuracy:.4f}"
        )
        mlflow.log_metric(f"{phase_name}_train_loss", train_loss, step=epoch + 1)
        mlflow.log_metric(f"{phase_name}_train_accuracy", train_accuracy, step=epoch + 1)
        mlflow.log_metric(f"{phase_name}_val_loss", validation_loss, step=epoch + 1)
        mlflow.log_metric(
            f"{phase_name}_val_accuracy",
            validation_accuracy,
            step=epoch + 1,
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_classes,
                    "validation_accuracy": validation_accuracy,
                    "phase": phase_name,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
            mlflow.log_metric("best_validation_accuracy", best_validation_accuracy)
            print(f"Saved new best model: {checkpoint_path}")

    return best_validation_accuracy


def main() -> None:
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    image_size = config["data"].get("image_size", 224)
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config["data"].get("num_workers", 0)
    head_learning_rate = config["training"].get("head_learning_rate", 0.001)
    head_epochs = config["training"].get("head_epochs", 3)
    finetune_learning_rate = config["training"].get("finetune_learning_rate", 0.0001)
    finetune_epochs = config["training"].get("finetune_epochs", 3)

    train_dataset = datasets.ImageFolder(
        root=processed_dir / "train",
        transform=get_train_transforms(image_size),
    )
    validation_dataset = datasets.ImageFolder(
        root=processed_dir / "validation",
        transform=get_eval_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_resnet18(
        num_classes=len(train_dataset.classes),
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=head_learning_rate,
    )
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "best_model.pt"
    best_validation_accuracy = 0.0

    print(f"Device: {device}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="resnet18_head_layer4_finetune"):
        mlflow.log_params(
            {
                "architecture": "resnet18",
                "num_classes": len(train_dataset.classes),
                "image_size": image_size,
                "batch_size": batch_size,
                "head_epochs": head_epochs,
                "head_learning_rate": head_learning_rate,
                "finetune_epochs": finetune_epochs,
                "finetune_learning_rate": finetune_learning_rate,
                "device": str(device),
            }
        )

        best_validation_accuracy = run_training_phase(
            phase_name="head",
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=head_epochs,
            checkpoint_path=checkpoint_path,
            train_classes=train_dataset.classes,
            best_validation_accuracy=best_validation_accuracy,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            "Loaded best head checkpoint before layer4 fine-tuning: "
            f"val_acc={checkpoint['validation_accuracy']:.4f}"
        )

        unfreeze_layer4(model)
        optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=finetune_learning_rate,
        )
        print(
            "Trainable parameters after unfreezing layer4: "
            f"{count_trainable_parameters(model)}"
        )

        run_training_phase(
            phase_name="layer4_finetune",
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=finetune_epochs,
            checkpoint_path=checkpoint_path,
            train_classes=train_dataset.classes,
            best_validation_accuracy=best_validation_accuracy,
        )

        mlflow.log_artifact(checkpoint_path)


if __name__ == "__main__":
    main()
