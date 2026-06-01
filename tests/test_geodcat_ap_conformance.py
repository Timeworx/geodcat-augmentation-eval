"""Automated GeoDCAT-AP 3.0 conformance gate (Task 3 / E1 of the eval brief).

For every entry in `eval/corpus/manifest.csv`, route through the *real*
`augment_ai_ready` or `build_legacy_skeleton`, validate against the
vendored SHACL shapes, and fail the test if the output produces any
`sh:Violation`. Warnings are reported but do not fail the gate — that
matches the paper's framing (Violations = the contribution's claim;
Warnings = the empirical landscape of real client data).

Skipped (not failed) when:
  * `pyshacl` / `rdflib` aren't installed (developer hasn't run
    `pip install -r requirements-dev.txt` yet);
  * the vendored shapes aren't present (run `python -m eval.vendor_shapes`
    on a workstation first).

Both conditions surface a clear human-readable skip reason rather than
silently passing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"
SHAPES_DIR = REPO_ROOT / "tests" / "fixtures" / "shacl" / "geodcat-ap-3.0" / "shapes"


# --- Skip conditions -------------------------------------------------------


def _missing_dev_deps_reason() -> str | None:
    try:
        import pyshacl  # noqa: F401
        import rdflib  # noqa: F401
    except ImportError as e:
        return (
            f"dev dependency missing ({e.name}); "
            f"run `pip install -r requirements-dev.txt`"
        )
    return None


def _missing_vendor_reason() -> str | None:
    if not SHAPES_DIR.exists() or not list(SHAPES_DIR.glob("*.ttl")):
        return (
            f"SHACL shapes not vendored under {SHAPES_DIR}; "
            f"run `python -m eval.vendor_shapes` on a workstation first"
        )
    return None


def _missing_manifest_reason() -> str | None:
    if not MANIFEST.exists():
        return (
            f"corpus manifest not present at {MANIFEST}; "
            f"run `python -m eval.build_corpus` first"
        )
    return None


_SKIP_REASON = (
    _missing_dev_deps_reason()
    or _missing_vendor_reason()
    or _missing_manifest_reason()
)


# --- Manifest -> parametrization ------------------------------------------


def _manifest_rows() -> list[dict]:
    if _SKIP_REASON or not MANIFEST.exists():
        return []
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


_ROWS = _manifest_rows()


def _ids(row: dict) -> str:
    return f"{row['origin']}::{row['expected_path']}::{row['id']}"


# --- The gate --------------------------------------------------------------


# Synthetic variants whose input DELIBERATELY violates GeoDCAT-AP 3.0 to
# stress a known shape friction. The augmenter's zero-loss invariant
# means the output inherits the violation — that's the intended design,
# not a regression. xfail keeps the cell on Table 1 but doesn't break CI.
_KNOWN_INPUT_VIOLATIONS = {
    "multi_geometry": (
        "multi_geometry input has two locn:geometry encodings as WKT "
        "literals where GeoDCAT-AP requires locn:Geometry nodes; the "
        "augmenter preserves both (zero-loss). Brief calls this out as "
        "the known multi-geometry SHACL friction."
    ),
}


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
@pytest.mark.parametrize("row", _ROWS, ids=[_ids(r) for r in _ROWS])
def test_geodcat_ap_conformance(row: dict, request) -> None:
    rid = row["id"]
    if rid in _KNOWN_INPUT_VIOLATIONS:
        request.node.add_marker(
            pytest.mark.xfail(reason=_KNOWN_INPUT_VIOLATIONS[rid], strict=True)
        )
    _assert_conforms(row)


def _assert_conforms(row: dict) -> None:
    """Every sidecar-produced output must conform (no `sh:Violation`).

    Warnings are collected into the test's recorded property bag so a
    failing-on-warnings run is easy to opt into later — but the
    contribution claim is Violation-clean.
    """
    # Imports inline so the module imports cleanly when deps are missing
    # (the file-level skipif still runs).
    from src.metadata.augmenter import augment_ai_ready
    from src.metadata.skeleton import build_legacy_skeleton
    from src.metadata.types import SegmentationType

    from tests.shacl_helpers import validate

    coco_url = f"https://example.org/coco/{row['id']}.json"
    expected = row["expected_path"]
    body = json.loads((REPO_ROOT / row["path"]).read_text(encoding="utf-8"))

    if expected == "ai_ready":
        output = augment_ai_ready(
            body,
            job_id=row["id"],
            segmentation_type=SegmentationType.INSTANCE,
            coco_access_url=coco_url,
        )
    elif expected == "legacy":
        output = build_legacy_skeleton(
            job_id=row["id"],
            segmentation_type=SegmentationType.INSTANCE,
            raw_uri=body.get("rawDataUri", ""),
            target_class=body.get("targetClass", "object"),
            coco_access_url=coco_url,
        )
    else:
        pytest.fail(f"unknown expected_path={expected!r} on row {row['id']}")

    result = validate(output)
    if not result.conforms:
        # Render the first few violations into the assertion message; the
        # full report is in `result.text` if a deeper look is needed.
        lines = ["GeoDCAT-AP 3.0 SHACL violations:"]
        for v in result.violations[:5]:
            lines.append(
                f"  - {v.source_shape} @ {v.focus_node}\n"
                f"      path: {v.path}\n"
                f"      msg : {v.message[:200]}"
            )
        if len(result.violations) > 5:
            lines.append(f"  … {len(result.violations) - 5} more")
        pytest.fail("\n".join(lines))


def test_corpus_manifest_present() -> None:
    """Sanity: the gate above requires the manifest. Surface that as a
    distinct, always-runnable test so CI tells the developer to build the
    corpus before running E1."""
    if _SKIP_REASON and "manifest" in _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    assert MANIFEST.exists(), _SKIP_REASON or "manifest missing"
