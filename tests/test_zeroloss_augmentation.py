"""Zero-loss augmentation CI gate (Task 4 / E2 of the eval brief).

For each AI-Ready manifest row, run the production augmenter and assert
that no client predicate disappears from the output *unless* it is in
the augmenter's explicit-replacement set:

    dct:identifier, dct:publisher, dcat:distribution

These three describe Timeworx's *new* dataset and are intentionally
rewritten. Everything else — title (carried under prov:wasDerivedFrom),
description (re-rooted as dct:type under prov:wasDerivedFrom), themes,
spatial, temporal, keywords, and any custom client predicate — must
survive. A disappearance outside that set is a real regression.

Also: the synthetic `custom_predicates` row carries four predicates
under https://client.example/internal# . They must ride through
end-to-end on the output dataset node — explicit assertion, not just
preservation_rate.

Skipped (not failed) when:
  * rdflib isn't installed;
  * the corpus manifest hasn't been built.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"

# Predicates the augmenter is allowed to drop without it counting as a
# regression. The other rewrites (@id, dct:title, dct:description) carry
# their original values forward under prov:wasDerivedFrom, so the diff
# logic in eval/run_e2_zeroloss.py already counts them as preserved.
_INTENTIONALLY_REPLACED = frozenset({
    "http://purl.org/dc/terms/identifier",
    "http://purl.org/dc/terms/publisher",
    "http://www.w3.org/ns/dcat#distribution",
})


# --- Skip conditions -------------------------------------------------------


def _missing_dev_deps_reason() -> str | None:
    try:
        import rdflib  # noqa: F401
    except ImportError as e:
        return (
            f"dev dependency missing ({e.name}); "
            f"run `pip install -r requirements-dev.txt`"
        )
    return None


def _missing_manifest_reason() -> str | None:
    if not MANIFEST.exists():
        return (
            f"corpus manifest not present at {MANIFEST}; "
            f"run `python -m eval.build_corpus` first"
        )
    return None


_SKIP_REASON = _missing_dev_deps_reason() or _missing_manifest_reason()


# --- Manifest -> parametrization ------------------------------------------


def _ai_ready_rows() -> list[dict]:
    if _SKIP_REASON or not MANIFEST.exists():
        return []
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["expected_path"] == "ai_ready"]


_ROWS = _ai_ready_rows()


def _ids(row: dict) -> str:
    return f"{row['origin']}::{row['id']}"


# --- The gate --------------------------------------------------------------


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
@pytest.mark.parametrize("row", _ROWS, ids=[_ids(r) for r in _ROWS])
def test_no_unintended_loss(row: dict) -> None:
    """The augmenter must not silently drop client predicates outside the
    explicit-replacement set. dct:identifier / dct:publisher /
    dcat:distribution may disappear (they describe Timeworx's new
    dataset); anything else lost is a regression.
    """
    from eval.run_e2_zeroloss import _diff, _run_augment

    input_doc = json.loads((REPO_ROOT / row["path"]).read_text(encoding="utf-8"))
    output = _run_augment(input_doc, row["id"])
    _, _, lost_predicates = _diff(input_doc, output, row_id=row["id"])

    unintended = sorted({p for p in lost_predicates if p not in _INTENTIONALLY_REPLACED})
    assert not unintended, (
        f"augmenter dropped non-replaceable client predicates on {row['id']!r}: "
        f"{unintended}"
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_custom_predicates_pass_through() -> None:
    """The four predicates under https://client.example/internal# on the
    synthetic `custom_predicates` row must survive on the augmented
    dataset node — that is the zero-loss invariant in its simplest form.
    """
    from rdflib import URIRef
    from eval.run_e2_zeroloss import (
        _CUSTOM_PREDS,
        _find_dataset_uri,
        _run_augment,
        _to_graph,
    )

    syn_row = next((r for r in _ROWS if r["id"] == "custom_predicates"), None)
    if syn_row is None:
        pytest.skip("custom_predicates synthetic row not in manifest")

    input_doc = json.loads((REPO_ROOT / syn_row["path"]).read_text(encoding="utf-8"))
    output = _run_augment(input_doc, syn_row["id"])
    out_g = _to_graph(output)
    root = _find_dataset_uri(out_g)
    assert root is not None, "no dcat:Dataset in augmented custom_predicates output"
    missing = [
        p for p in _CUSTOM_PREDS
        if next(out_g.objects(root, URIRef(p)), None) is None
    ]
    assert not missing, f"augmenter dropped custom predicates: {missing}"


def test_e2_corpus_manifest_present() -> None:
    """Always-runnable sanity: surface a missing-corpus state distinctly
    instead of silently emptying the parametrization above.
    """
    if _SKIP_REASON and "manifest" in _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    assert MANIFEST.exists(), _SKIP_REASON or "manifest missing"
