from pathlib import Path

import requests

from src.config import PROJECT_ROOT


def send_image(api_url: str, image_path: Path) -> None:
    with image_path.open("rb") as file:
        response = requests.post(
            f"{api_url}/predict",
            files={"file": (image_path.name, file, "image/jpeg")},
            timeout=30,
        )

    response.raise_for_status()
    result = response.json()

    print(
        f"{image_path.name}: "
        f"{result['prediction']} "
        f"confidence={result['confidence']:.4f}"
    )


def main() -> None:
    api_url = "http://127.0.0.1:8001"
    image_dir = PROJECT_ROOT / "data" / "production" / "drifted"
    image_paths = sorted(image_dir.rglob("*.jpg"))

    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    for image_path in image_paths:
        send_image(api_url, image_path)

    print(f"Sent {len(image_paths)} images to {api_url}")


if __name__ == "__main__":
    main()
