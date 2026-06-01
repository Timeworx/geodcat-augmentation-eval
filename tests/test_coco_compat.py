"""COCO compatibility + semantic round-trip CI gate (Task 5 / E3).

For each sidecar COCO sample (the committed sample-coco.json fixture plus
samples synthesized from the AI-Ready synthetic corpus), assert:

  * `pycocotools.coco.COCO` parses without error and exposes the
    standard category fields (`id`, `name`, `supercategory`).
  * Plain `json.load` sees `semantic_uri` on at least one category — the
    contribution claim is that the sidecar's added field rides through
    the COCO file untouched.
  * For samples paired with an augmented GeoDCAT-AP graph: the set of
    AGROVOC URIs in `dcat:theme` whose `skos:prefLabel` matches a COCO
    category name equals the set of `categories[].semantic_uri`. That
    is the round-trip claim from §7.4 of the paper.

Skipped (not failed) when:
  * `pycocotools` is not installed (run
    `pip install -r requirements-dev.txt`);
  * the committed COCO fixture is missing.

torchvision is tested by the runner but not the gate — pulling torch
into CI is heavy, and the `pycocotools` parse already covers the
critical "doesn't crash on semantic_uri" question.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_COCO = REPO_ROOT / "tests" / "fixtures" / "sample-coco.json"
FIXTURE_INPUT = REPO_ROOT / "tests" / "fixtures" / "input-geodcat.jsonld"


def _missing_pycocotools_reason() -> str | None:
    try:
        import pycocotools.coco  # noqa: F401
    except ImportError as e:
        return (
            f"dev dependency missing ({e.name}); "
            f"run `pip install -r requirements-dev.txt`"
        )
    return None


def _missing_fixture_reason() -> str | None:
    if not FIXTURE_COCO.exists():
        return f"COCO fixture missing at {FIXTURE_COCO}"
    return None


_SKIP_REASON = _missing_pycocotools_reason() or _missing_fixture_reason()


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_pycocotools_parses_fixture_with_semantic_uri(tmp_path: Path) -> None:
    """The de-facto standard COCO consumer must parse our extended file
    and surface the standard category fields. semantic_uri is an added
    field, not a replacement — pycocotools stores it on the category
    dict alongside id/name/supercategory.
    """
    from pycocotools.coco import COCO

    coco_path = tmp_path / "sample.json"
    coco_path.write_text(FIXTURE_COCO.read_text(encoding="utf-8"))
    api = COCO(str(coco_path))
    cats = api.loadCats(api.getCatIds())
    assert cats, "pycocotools loaded the fixture but found no categories"
    for c in cats:
        for required in ("id", "name", "supercategory"):
            assert required in c, f"category {c} missing standard field {required!r}"


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_semantic_uri_rides_through_plain_json() -> None:
    """The simplest consumer (`json.load`) must see `semantic_uri` on at
    least one category. This is the broadest-surface check — if a
    consumer that does nothing more than parse JSON misses it, the
    extension isn't compatible.
    """
    doc = json.loads(FIXTURE_COCO.read_text(encoding="utf-8"))
    cats = doc.get("categories", [])
    assert any("semantic_uri" in c for c in cats), (
        "no category in the fixture carries `semantic_uri` — the COCO "
        "proxy injection is the sidecer's claim, the fixture must reflect it"
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_semantic_uri_is_subset_of_agrovoc_themes() -> None:
    """Every `categories[].semantic_uri` the sidecar emits must trace
    back to an AGROVOC URI present in the input's `dcat:theme` — the
    sidecar never invents semantic URIs. The committed sample-coco.json
    was hand-curated against an input whose AGROVOC label is "maize"
    while the COCO category name is "maize plant", so strict
    set-equality is not the right check here; subset-inclusion is.

    Strict set-equality is exercised by the synthetic pair below, where
    the sidecar generates both COCO categories and the
    `class_to_agrovoc` map from the same theme source.
    """
    from eval.run_e3_coco import _extract_agrovoc_themes

    if not FIXTURE_INPUT.exists():
        pytest.skip(f"input fixture missing at {FIXTURE_INPUT}")

    coco = json.loads(FIXTURE_COCO.read_text(encoding="utf-8"))
    input_doc = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    dataset = input_doc["@graph"][0]
    agrovoc_themes = set(_extract_agrovoc_themes(dataset.get("dcat:theme", [])).values())
    semantic_uris = {
        c["semantic_uri"]
        for c in coco.get("categories", [])
        if isinstance(c, dict) and isinstance(c.get("semantic_uri"), str)
    }
    orphans = sorted(semantic_uris - agrovoc_themes)
    assert not orphans, (
        f"broker COCO carries semantic_uri values absent from input dcat:theme "
        f"AGROVOC URIs: {orphans}"
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_broker_generated_pair_round_trips() -> None:
    """When the sidecar builds *both* the COCO category set and the
    augmented GeoDCAT-AP from the same input themes, the AGROVOC URI
    sets must be exactly equal — that's the strict round-trip claim
    in §7.4 of the paper. This synthesizes the pair on the fly.
    """
    from src.metadata.augmenter import augment_ai_ready
    from src.metadata.types import SegmentationType

    from eval.run_e3_coco import _extract_agrovoc_themes, _roundtrip, _stub_coco
    from src.metadata.coco import inject_semantic_uris

    if not FIXTURE_INPUT.exists():
        pytest.skip(f"input fixture missing at {FIXTURE_INPUT}")

    input_doc = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    dataset = input_doc["@graph"][0]
    class_to_agrovoc = _extract_agrovoc_themes(dataset.get("dcat:theme", []))
    assert class_to_agrovoc, "input fixture has no AGROVOC themes to round-trip"

    coco = inject_semantic_uris(_stub_coco(class_to_agrovoc), class_to_agrovoc)
    augmented = augment_ai_ready(
        input_doc,
        job_id="fixture-maize",
        segmentation_type=SegmentationType.INSTANCE,
        coco_access_url="https://example.org/coco/fixture-maize.json",
    )
    row = _roundtrip("fixture_maize_generated", coco, augmented)
    assert row.set_equal == "yes", (
        f"round-trip URI set mismatch: theme→coco missing "
        f"{row.missing_in_coco!r}, coco→theme extra {row.extra_in_coco!r}"
    )
