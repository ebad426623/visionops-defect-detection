from pathlib import Path
import json
import random
import shutil

from src.config import PROJECT_ROOT, load_config


def get_class_dirs(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_dir()])


def get_images(class_dir: Path, allowed_extensions: list[str]) -> list[Path]:
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ]
    )


def create_processed_dirs(processed_dir: Path, class_names: list[str]) -> None:
    for split in ["train", "validation", "test"]:
        for class_name in class_names:
            output_dir = processed_dir / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)


def reset_processed_dir(processed_dir: Path) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], output_dir: Path) -> None:
    for image_path in images:
        destination = output_dir / image_path.name
        shutil.copy2(image_path, destination)


def split_images(
    images: list[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    shuffled_images = images.copy()
    random.Random(seed).shuffle(shuffled_images)

    validation_count = int(len(shuffled_images) * validation_fraction)

    validation_images = shuffled_images[:validation_count]
    test_images = shuffled_images[validation_count:]

    return validation_images, test_images


def main() -> None:
    config = load_config()

    train_dir = PROJECT_ROOT / config["data"]["raw_train_dir"]
    validation_dir = PROJECT_ROOT / config["data"]["raw_validation_dir"]
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    allowed_extensions = config["data"]["allowed_extensions"]
    validation_fraction = config["data"]["validation_fraction_from_original_validation"]
    seed = config["project"]["seed"]

    train_classes = get_class_dirs(train_dir)
    validation_classes = get_class_dirs(validation_dir)

    if [path.name for path in train_classes] != [
        path.name for path in validation_classes
    ]:
        raise ValueError("Train and validation classes do not match.")

    class_names = [path.name for path in train_classes]

    reset_processed_dir(processed_dir)
    create_processed_dirs(processed_dir, class_names)

    total_copied = 0
    total_validation_copied = 0
    total_test_copied = 0
    class_counts = {}

    for class_index, class_name in enumerate(class_names):
        source_dir = train_dir / class_name
        output_dir = processed_dir / "train" / class_name

        images = get_images(source_dir, allowed_extensions)
        copy_images(images, output_dir)

        total_copied += len(images)
        print(f"Copied train/{class_name}: {len(images)} images")

        original_validation_dir = validation_dir / class_name
        original_validation_images = get_images(
            original_validation_dir, allowed_extensions
        )

        validation_images, test_images = split_images(
            original_validation_images,
            validation_fraction,
            seed + class_index,
        )

        validation_output_dir = processed_dir / "validation" / class_name
        test_output_dir = processed_dir / "test" / class_name

        copy_images(validation_images, validation_output_dir)
        copy_images(test_images, test_output_dir)

        total_validation_copied += len(validation_images)
        total_test_copied += len(test_images)

        class_counts[class_name] = {
            "train": len(images),
            "validation": len(validation_images),
            "test": len(test_images),
        }

        print(f"Copied validation/{class_name}: {len(validation_images)} images")
        print(f"Copied test/{class_name}: {len(test_images)} images")

    print(f"Total train images copied: {total_copied}")
    print(f"Total validation images copied: {total_validation_copied}")
    print(f"Total test images copied: {total_test_copied}")

    report = {
        "train": total_copied,
        "validation": total_validation_copied,
        "test": total_test_copied,
        "classes": class_counts,
    }

    report_path = processed_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote split report: {report_path}")


if __name__ == "__main__":
    main()
