import json

from src.config import PROJECT_ROOT, load_config


def main() -> None:
    config = load_config()

    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    minimum_macro_f1 = config.get("evaluation", {}).get("minimum_macro_f1", 0.80)
    macro_f1 = metrics["macro_f1"]

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Minimum required macro F1: {minimum_macro_f1:.4f}")

    if macro_f1 < minimum_macro_f1:
        raise SystemExit("Quality gate failed.")

    print("Quality gate passed.")


if __name__ == "__main__":
    main()
