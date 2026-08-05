import json
from pathlib import Path

from src.config import PROJECT_ROOT, load_config


def run_quality_gate(metrics_path: Path, minimum_macro_f1: float) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    macro_f1 = metrics["macro_f1"]

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Minimum required macro F1: {minimum_macro_f1:.4f}")

    if macro_f1 < minimum_macro_f1:
        raise SystemExit("Quality gate failed.")

    print("Quality gate passed.")


def main() -> None:
    config = load_config()

    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    minimum_macro_f1 = config.get("evaluation", {}).get("minimum_macro_f1", 0.80)

    run_quality_gate(metrics_path, minimum_macro_f1)


if __name__ == "__main__":
    main()
