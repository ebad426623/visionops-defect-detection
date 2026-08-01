from pathlib import Path
from PIL import Image
from src.config import PROJECT_ROOT, load_config


def get_class_dirs(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.is_dir()])


def count_images(class_dir: Path, allowed_extensions: list[str]) -> int:
    count = 0

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            count += 1

    return count


def verify_images(class_dir: Path, allowed_extensions: list[str]) -> list[Path]:
    bad_images = []

    for path in class_dir.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_extensions:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                bad_images.append(path)

    return bad_images


def check_same_classes(train_classes: list[Path], validation_classes: list[Path]) -> None:
    train_names = [path.name for path in train_classes]
    validation_names = [path.name for path in validation_classes]

    if train_names != validation_names:
        raise ValueError(
            f"Train and validation classes do not match: "
            f"{train_names} != {validation_names}"
        )


def main() -> None:
    config = load_config()
    train_dir = PROJECT_ROOT / config["data"]["raw_train_dir"]
    validation_dir = PROJECT_ROOT / config["data"]["raw_validation_dir"]
    allowed_extensions = config["data"]["allowed_extensions"]

    train_classes = get_class_dirs(train_dir)
    validation_classes = get_class_dirs(validation_dir)

    check_same_classes(train_classes, validation_classes)

    total_train_images = 0
    total_validation_images = 0
    bad_images = []

    print("Train image counts:")
    for class_dir in train_classes:
        image_count = count_images(class_dir, allowed_extensions)

        if image_count == 0:
            raise ValueError(f"Class folder is empty: {class_dir}")

        bad_images.extend(verify_images(class_dir, allowed_extensions))

        total_train_images += image_count
        print(f"- {class_dir.name}: {image_count}")

    print("Validation image counts:")
    for class_dir in validation_classes:
        image_count = count_images(class_dir, allowed_extensions)

        if image_count == 0:
            raise ValueError(f"Class folder is empty: {class_dir}")

        bad_images.extend(verify_images(class_dir, allowed_extensions))

        total_validation_images += image_count
        print(f"- {class_dir.name}: {image_count}")

    print(f"Total train images: {total_train_images}")
    print(f"Total validation images: {total_validation_images}")
    print(f"Total images: {total_train_images + total_validation_images}")

    if bad_images:
        print("Bad images:")
        for path in bad_images:
            print(f"- {path}")
        raise ValueError(f"Found {len(bad_images)} bad images.")

    print("All images are readable.")


if __name__ == "__main__":
    main()
