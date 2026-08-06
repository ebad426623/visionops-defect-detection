from pathlib import Path

from PIL import Image, ImageEnhance

from src.config import PROJECT_ROOT


def darken_image(source_path: Path, output_path: Path, factor: float = 0.45) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        darkened = ImageEnhance.Brightness(image).enhance(factor)
        darkened.save(output_path)


def main() -> None:
    source_dir = PROJECT_ROOT / "data" / "processed" / "test"
    output_dir = PROJECT_ROOT / "data" / "production" / "drifted"

    image_paths = sorted(source_dir.rglob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No test images found in: {source_dir}")

    copied = 0
    for image_path in image_paths:
        class_name = image_path.parent.name
        output_path = output_dir / class_name / image_path.name
        darken_image(image_path, output_path)
        copied += 1

    print(f"Created drifted images: {copied}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
