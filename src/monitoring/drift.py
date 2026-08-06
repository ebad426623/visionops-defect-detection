import argparse
import json
from collections import Counter
from pathlib import Path

from src.config import PROJECT_ROOT


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Analyze only the most recent N prediction records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_log_path = PROJECT_ROOT / "artifacts" / "predictions.jsonl"
    report_path = PROJECT_ROOT / "artifacts" / "drift_report.json"

    records = load_jsonl(prediction_log_path)
    if args.last_n is not None:
        records = records[-args.last_n :]

    confidences = [record["confidence"] for record in records]
    brightness_values = [
        record["image_stats"]["mean_brightness"]
        for record in records
        if "image_stats" in record
    ]
    predicted_classes = [record["prediction"] for record in records]
    class_counts = Counter(predicted_classes)

    average_confidence = average(confidences)
    average_brightness = average(brightness_values)

    confidence_drift = average_confidence < 0.90
    brightness_drift = average_brightness < 100
    class_distribution_drift = (
        max(class_counts.values()) / len(predicted_classes) > 0.50
    )
    dataset_drift = confidence_drift or brightness_drift or class_distribution_drift

    report = {
        "dataset_drift": dataset_drift,
        "confidence_drift": confidence_drift,
        "brightness_drift": brightness_drift,
        "class_distribution_drift": class_distribution_drift,
        "num_predictions": len(records),
        "average_confidence": average_confidence,
        "average_brightness": average_brightness,
        "class_distribution": dict(class_counts),
        "thresholds": {
            "minimum_average_confidence": 0.90,
            "minimum_average_brightness": 100,
            "maximum_single_class_fraction": 0.50,
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset drift: {dataset_drift}")
    print(f"Average confidence: {average_confidence:.4f}")
    print(f"Average brightness: {average_brightness:.2f}")
    print(f"Wrote drift report: {report_path}")


if __name__ == "__main__":
    main()
