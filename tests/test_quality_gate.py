import json
from pathlib import Path

import pytest

from src.evaluation.quality_gate import run_quality_gate


def test_quality_gate_passes(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"macro_f1": 0.90}), encoding="utf-8")

    run_quality_gate(metrics_path, minimum_macro_f1=0.80)


def test_quality_gate_fails(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"macro_f1": 0.70}), encoding="utf-8")

    with pytest.raises(SystemExit):
        run_quality_gate(metrics_path, minimum_macro_f1=0.80)
