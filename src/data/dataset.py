from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import PROJECT_ROOT, load_config


def get_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def main() -> None:
    config = load_config()

    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    image_size = config["data"].get("image_size", 224)
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config["data"].get("num_workers", 0)

    train_dataset = datasets.ImageFolder(
        root=processed_dir / "train",
        transform=get_train_transforms(image_size),
    )

    validation_dataset = datasets.ImageFolder(
        root=processed_dir / "validation",
        transform=get_eval_transforms(image_size),
    )

    test_dataset = datasets.ImageFolder(
        root=processed_dir / "test",
        transform=get_eval_transforms(image_size),
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    images, labels = next(iter(train_loader))

    print(f"One image batch shape: {images.shape}")
    print(f"One label batch shape: {labels.shape}")


if __name__ == "__main__":
    main()
