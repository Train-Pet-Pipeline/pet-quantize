"""NoopConverter .run() contract tests."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pet_quantize.plugins.converters.noop import NoopConverter
from pet_schema.model_card import ModelCard


def _make_card(card_id: str = "test-card-1") -> ModelCard:
    """Build a minimal valid ModelCard (all ModelCard required fields, extra='forbid')."""
    return ModelCard(
        id=card_id,
        version="1.0.0",
        modality="vision",
        task="classification",
        arch="test-arch",
        training_recipe="test-recipe",
        hydra_config_sha="a" * 40,
        git_shas={"pet_quantize": "b" * 40},
        dataset_versions={"test_dataset": "v1"},
        checkpoint_uri="/tmp/ckpt",
        metrics={"accuracy": 0.5},
        gate_status="passed",
        trained_at=datetime.now(UTC),
        trained_by="test",
    )


def test_noop_appends_edge_artifact() -> None:
    card = _make_card()
    out = NoopConverter().run(card, recipe=None)
    assert len(out.edge_artifacts) == 1
    assert len(out.edge_artifacts[0].sha256) == 64
    assert out.edge_artifacts[0].format == "onnx"
    assert out.edge_artifacts[0].target_hardware == ["cpu"]


def test_noop_is_deterministic_per_card_id() -> None:
    """Same input id → same sha256 (supports orchestrator resume-from-cache)."""
    a = NoopConverter().run(_make_card("stable"), recipe=None).edge_artifacts[0].sha256
    b = NoopConverter().run(_make_card("stable"), recipe=None).edge_artifacts[0].sha256
    assert a == b


def test_noop_varies_by_card_id() -> None:
    """Different input id → different sha256."""
    a = NoopConverter().run(_make_card("one"), recipe=None).edge_artifacts[0].sha256
    b = NoopConverter().run(_make_card("two"), recipe=None).edge_artifacts[0].sha256
    assert a != b
