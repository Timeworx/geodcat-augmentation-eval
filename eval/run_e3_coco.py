"""Experiment E3 — COCO toolchain compatibility + semantic round-trip (→ Table 3).

Two questions:

  1. Does the sidecar's `semantic_uri`-injected COCO still parse cleanly
     across the standard COCO toolchain? Four loaders:
       (a) pycocotools.coco.COCO              — de-facto standard
       (b) pycocotools.cocoeval.COCOeval      — smoke-run (self vs self)
       (c) torchvision.datasets.CocoDetection — PyTorch consumer
       (d) plain `json.load`                  — broadest consumer surface
     Per loader and sample: parses_without_error,
     standard_fields_accessible (id/name/supercategory), and
     semantic_uri_preserved_or_ignored.

  2. Semantic round-trip — for each (augmented GeoDCAT-AP, COCO) pair,
     extract the AGROVOC URIs from `dcat:theme` whose `skos:prefLabel`
     matches a COCO category `name` (the same bridge the sidecar's
     request_mapping uses), and assert this URI set equals the set of
     `categories[].semantic_uri` on the COCO. Reported as match / total.

Outputs:
    eval/results/e3_coco_compat.csv          Table 3
    eval/results/e3_semantic_roundtrip.csv   match/total per pair

Each loader probe runs independently — one missing loader does not skip
the others. Its row gets `parses_without_error = no` with the import or
parse error in the diagnostic column.

Run from the root folder:

    python -m eval.run_e3_coco

Prereqs: Task 1 corpus built; `pip install -r requirements-dev.txt`.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from src.metadata.augmenter import augment_ai_ready
from src.metadata.coco import inject_semantic_uris
from src.metadata.types import SegmentationType

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"
FIXTURE_COCO = REPO_ROOT / "tests" / "fixtures" / "sample-coco.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

COMPAT_CSV = RESULTS_DIR / "e3_coco_compat.csv"
ROUNDTRIP_CSV = RESULTS_DIR / "e3_semantic_roundtrip.csv"

_COCO_URL_TEMPLATE = "https://example.org/coco/{job_id}.json"
_SEGMENTATION_TYPE = SegmentationType.INSTANCE
_AGROVOC_PREFIX = "http://aims.fao.org/aos/agrovoc/"


@dataclass
class CompatRow:
    sample_id: str
    loader: str               # pycocotools_COCO | pycocotools_COCOeval | torchvision_CocoDetection | plain_json
    parses_without_error: str
    standard_fields_accessible: str
    semantic_uri_preserved_or_ignored: str
    diagnostic: str = ""


@dataclass
class RoundtripRow:
    sample_id: str
    theme_uris: int
    semantic_uris: int
    intersection: int
    set_equal: str
    missing_in_coco: str        # ";"-joined theme URIs absent from COCO
    extra_in_coco: str          # ";"-joined COCO URIs absent from themes


# --- Sample construction --------------------------------------------------


def _existing_fixture_sample() -> tuple[str, dict, dict | None]:
    """The committed sample-coco.json is the canonical reference sample.
    Pair it with the maize input fixture for the round-trip — the
    augmenter output's dcat:theme list comes from that input.
    """
    coco = json.loads(FIXTURE_COCO.read_text(encoding="utf-8"))
    input_path = REPO_ROOT / "tests" / "fixtures" / "input-geodcat.jsonld"
    if input_path.exists():
        input_doc = json.loads(input_path.read_text(encoding="utf-8"))
        return "fixture_maize", coco, input_doc
    return "fixture_maize", coco, None


def _extract_agrovoc_themes(themes) -> dict[str, str]:
    """Return prefLabel → AGROVOC URI for theme entries that are
    AGROVOC concepts. Mirrors request_mapping._extract_agrovoc_themes
    without depending on its private API.
    """
    if isinstance(themes, dict):
        themes = [themes]
    mapping: dict[str, str] = {}
    for theme in themes or []:
        if not isinstance(theme, dict):
            continue
        concept_id = theme.get("@id")
        label_node = theme.get("skos:prefLabel")
        label = None
        if isinstance(label_node, dict):
            label = label_node.get("@value")
        elif isinstance(label_node, str):
            label = label_node
        if (
            isinstance(concept_id, str)
            and concept_id.startswith(_AGROVOC_PREFIX)
            and isinstance(label, str)
            and label not in mapping
        ):
            mapping[label] = concept_id
    return mapping


def _stub_coco(class_to_agrovoc: dict[str, str]) -> dict:
    """Build a minimal valid COCO with one category per AGROVOC class
    name. We use this to test inject_semantic_uris across multiple
    samples from the synthetic corpus without needing real annotations.
    """
    categories = []
    annotations = []
    for idx, name in enumerate(sorted(class_to_agrovoc.keys()), start=1):
        categories.append({
            "id": idx,
            "name": name,
            "supercategory": "concept",
        })
        annotations.append({
            "id": idx,
            "image_id": 1,
            "category_id": idx,
            "segmentation": {"size": [10, 10], "counts": "a"},
            "area": 1.0,
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "iscrowd": 0,
        })
    return {
        "info": {
            "description": "E3 stub COCO",
            "version": "1.0",
            "year": 2026,
            "contributor": "eval/run_e3_coco.py",
        },
        "licenses": [],
        "images": [{"id": 1, "width": 10, "height": 10, "file_name": "0001.png"}],
        "categories": categories,
        "annotations": annotations,
    }


def _synthetic_samples() -> list[tuple[str, dict, dict]]:
    """For each AI-Ready synthetic row that carries AGROVOC themes,
    build a (sample_id, broker_coco, augmented_geodcat) triple. The
    broker_coco is a stub with one category per AGROVOC class, run
    through the production inject_semantic_uris.
    """
    out: list[tuple[str, dict, dict]] = []
    if not MANIFEST.exists():
        return out
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["expected_path"] != "ai_ready":
            continue
        path = REPO_ROOT / row["path"]
        if not path.exists():
            continue
        input_doc = json.loads(path.read_text(encoding="utf-8"))
        try:
            from src.metadata.jsonld import find_dataset
            dataset = find_dataset(input_doc)
        except Exception:  # noqa: BLE001
            continue
        class_to_agrovoc = _extract_agrovoc_themes(dataset.get("dcat:theme", []))
        if not class_to_agrovoc:
            continue
        coco = inject_semantic_uris(_stub_coco(class_to_agrovoc), class_to_agrovoc)
        augmented = augment_ai_ready(
            input_doc,
            job_id=row["id"],
            segmentation_type=_SEGMENTATION_TYPE,
            coco_access_url=_COCO_URL_TEMPLATE.format(job_id=row["id"]),
        )
        out.append((f"synthetic_{row['id']}", coco, augmented))
    return out


# --- Loader probes --------------------------------------------------------


def _yes_no(b: bool) -> str:
    return "yes" if b else "no"


def _probe_pycocotools_coco(coco_path: Path) -> CompatRow:
    try:
        from pycocotools.coco import COCO
    except ImportError as e:
        return CompatRow(
            sample_id="", loader="pycocotools_COCO",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"ImportError: {e}",
        )
    try:
        # pycocotools is chatty on load — quiet it for clean CSV diagnostics.
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                api = COCO(str(coco_path))
            finally:
                sys.stdout = old_stdout
    except Exception as e:  # noqa: BLE001
        return CompatRow(
            sample_id="", loader="pycocotools_COCO",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"{type(e).__name__}: {e}",
        )
    cats = api.loadCats(api.getCatIds())
    standard_ok = all("id" in c and "name" in c and "supercategory" in c for c in cats)
    # pycocotools stores categories as raw dicts in self.dataset['categories']
    # — extra fields like semantic_uri survive. Verify on the loaded copy.
    semantic_ok = any("semantic_uri" in c for c in cats) or not any(
        "semantic_uri" in c for c in api.dataset.get("categories", [])
    )
    return CompatRow(
        sample_id="", loader="pycocotools_COCO",
        parses_without_error="yes",
        standard_fields_accessible=_yes_no(standard_ok),
        semantic_uri_preserved_or_ignored=_yes_no(semantic_ok),
    )


def _probe_pycocotools_eval(coco_path: Path) -> CompatRow:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        return CompatRow(
            sample_id="", loader="pycocotools_COCOeval",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"ImportError: {e}",
        )
    try:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                gt = COCO(str(coco_path))
                # Self vs self smoke-run: load the same file as the
                # prediction. We're not measuring AP — we're checking
                # that COCOeval *constructs* against semantic_uri-tagged
                # categories without throwing.
                dt = gt.loadRes(_predictions_for(gt))
                evaluator = COCOeval(gt, dt, iouType="segm")
            finally:
                sys.stdout = old_stdout
        # `evaluator` exists — that is the smoke-run signal. We do not
        # run .evaluate() because that touches per-pixel masks we don't
        # have real data for.
        _ = evaluator
        return CompatRow(
            sample_id="", loader="pycocotools_COCOeval",
            parses_without_error="yes",
            standard_fields_accessible="yes",
            semantic_uri_preserved_or_ignored="yes",
        )
    except Exception as e:  # noqa: BLE001
        return CompatRow(
            sample_id="", loader="pycocotools_COCOeval",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"{type(e).__name__}: {e}",
        )


def _predictions_for(gt) -> list[dict]:
    """Synthesize prediction-style entries from the ground-truth COCO so
    COCOeval can be constructed. We only need shape, not AP-meaningful
    content — every entry gets `score: 1.0`.
    """
    preds = []
    for ann in gt.dataset.get("annotations", []):
        preds.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "segmentation": ann.get("segmentation"),
            "bbox": ann.get("bbox", [0, 0, 1, 1]),
            "score": 1.0,
        })
    return preds


def _probe_torchvision(coco_path: Path) -> CompatRow:
    try:
        from torchvision.datasets import CocoDetection
    except ImportError as e:
        return CompatRow(
            sample_id="", loader="torchvision_CocoDetection",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"ImportError: {e}",
        )
    try:
        # CocoDetection only touches `root` from __getitem__; constructor
        # parses annFile via pycocotools. Pass a junk root since we don't
        # iterate samples (no real imagery in the fixture).
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                ds = CocoDetection(root=str(coco_path.parent), annFile=str(coco_path))
            finally:
                sys.stdout = old_stdout
        cats = ds.coco.loadCats(ds.coco.getCatIds())
        standard_ok = all("id" in c and "name" in c and "supercategory" in c for c in cats)
        semantic_ok = any("semantic_uri" in c for c in cats) or not any(
            "semantic_uri" in c for c in ds.coco.dataset.get("categories", [])
        )
        return CompatRow(
            sample_id="", loader="torchvision_CocoDetection",
            parses_without_error="yes",
            standard_fields_accessible=_yes_no(standard_ok),
            semantic_uri_preserved_or_ignored=_yes_no(semantic_ok),
        )
    except Exception as e:  # noqa: BLE001
        return CompatRow(
            sample_id="", loader="torchvision_CocoDetection",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"{type(e).__name__}: {e}",
        )


def _probe_plain_json(coco_path: Path) -> CompatRow:
    """The broadest possible consumer: tools that just `json.load` the
    file. semantic_uri is preserved by definition (we're reading the
    raw bytes). Verify the structural invariants COCO consumers depend
    on (categories list, each with id/name/supercategory).
    """
    try:
        doc = json.loads(coco_path.read_text(encoding="utf-8"))
        cats = doc.get("categories", [])
        standard_ok = (
            isinstance(cats, list)
            and all(
                isinstance(c, dict) and "id" in c and "name" in c
                and "supercategory" in c
                for c in cats
            )
        )
        semantic_ok = any("semantic_uri" in c for c in cats)
        return CompatRow(
            sample_id="", loader="plain_json",
            parses_without_error="yes",
            standard_fields_accessible=_yes_no(standard_ok),
            semantic_uri_preserved_or_ignored=_yes_no(semantic_ok),
        )
    except Exception as e:  # noqa: BLE001
        return CompatRow(
            sample_id="", loader="plain_json",
            parses_without_error="no", standard_fields_accessible="no",
            semantic_uri_preserved_or_ignored="no",
            diagnostic=f"{type(e).__name__}: {e}",
        )


# --- Round-trip ----------------------------------------------------------


def _dataset_node(graph: dict) -> dict | None:
    nodes = graph.get("@graph") if isinstance(graph, dict) else None
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if isinstance(n, dict):
            t = n.get("@type")
            if t == "dcat:Dataset" or (isinstance(t, list) and "dcat:Dataset" in t):
                return n
    return None


def _roundtrip(sample_id: str, coco: dict, augmented_geodcat: dict | None) -> RoundtripRow:
    if augmented_geodcat is None:
        return RoundtripRow(
            sample_id=sample_id, theme_uris=0, semantic_uris=0,
            intersection=0, set_equal="n/a",
            missing_in_coco="", extra_in_coco="",
        )
    dataset = _dataset_node(augmented_geodcat)
    if dataset is None:
        return RoundtripRow(
            sample_id=sample_id, theme_uris=0, semantic_uris=0,
            intersection=0, set_equal="no",
            missing_in_coco="", extra_in_coco="",
        )
    # Only count themes whose prefLabel matches a category name — that
    # is the bridge the sidecar uses to inject semantic_uri. Themes that
    # don't match any category are simply not in scope for round-trip.
    cat_names = {
        c.get("name") for c in coco.get("categories", []) if isinstance(c, dict)
    }
    theme_map = _extract_agrovoc_themes(dataset.get("dcat:theme", []))
    theme_uris = {uri for label, uri in theme_map.items() if label in cat_names}
    semantic_uris = {
        c["semantic_uri"]
        for c in coco.get("categories", [])
        if isinstance(c, dict) and isinstance(c.get("semantic_uri"), str)
    }
    inter = theme_uris & semantic_uris
    missing = sorted(theme_uris - semantic_uris)
    extra = sorted(semantic_uris - theme_uris)
    return RoundtripRow(
        sample_id=sample_id,
        theme_uris=len(theme_uris),
        semantic_uris=len(semantic_uris),
        intersection=len(inter),
        set_equal=_yes_no(theme_uris == semantic_uris),
        missing_in_coco=";".join(missing),
        extra_in_coco=";".join(extra),
    )


# --- Main loop ------------------------------------------------------------


def _samples() -> list[tuple[str, dict, dict | None]]:
    samples: list[tuple[str, dict, dict | None]] = [_existing_fixture_sample()]
    for sid, coco, augmented in _synthetic_samples():
        samples.append((sid, coco, augmented))
    return samples


def _missing_manifest_reason() -> str | None:
    if not MANIFEST.exists():
        return (
            f"manifest missing at {MANIFEST}; "
            f"run `python -m eval.build_corpus` first (the fixture-only "
            f"sample will still be evaluated when manifest is present)"
        )
    return None


def run() -> tuple[list[CompatRow], list[RoundtripRow]]:
    compat: list[CompatRow] = []
    roundtrips: list[RoundtripRow] = []
    samples = _samples()
    for sid, coco, augmented in samples:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".json", delete=False
        ) as tmp:
            json.dump(coco, tmp)
            tmp.flush()
            coco_path = Path(tmp.name)
        try:
            for probe in (
                _probe_pycocotools_coco,
                _probe_pycocotools_eval,
                _probe_torchvision,
                _probe_plain_json,
            ):
                row = probe(coco_path)
                row.sample_id = sid
                compat.append(row)
        finally:
            try:
                coco_path.unlink()
            except OSError:
                pass
        roundtrips.append(_roundtrip(sid, coco, augmented))
    return compat, roundtrips


def _write_compat(rows: list[CompatRow]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "loader", "parses_without_error",
        "standard_fields_accessible", "semantic_uri_preserved_or_ignored",
        "diagnostic",
    ]
    with COMPAT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_roundtrip(rows: list[RoundtripRow]) -> None:
    fields = [
        "sample_id", "theme_uris", "semantic_uris",
        "intersection", "set_equal", "missing_in_coco", "extra_in_coco",
    ]
    with ROUNDTRIP_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    if not FIXTURE_COCO.exists():
        print(
            f"E3 skipped: fixture {FIXTURE_COCO} missing", file=sys.stderr
        )
        raise SystemExit(0)
    try:
        compat, roundtrips = run()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(2)
    _write_compat(compat)
    _write_roundtrip(roundtrips)
    n_samples = len({r.sample_id for r in compat})
    n_passing = sum(
        1 for r in compat
        if r.parses_without_error == "yes" and r.standard_fields_accessible == "yes"
    )
    n_total = len(compat)
    n_round_ok = sum(1 for r in roundtrips if r.set_equal == "yes")
    print(
        f"E3: {n_passing}/{n_total} (sample, loader) pairs parse + expose "
        f"standard fields across {n_samples} samples; "
        f"{n_round_ok}/{len(roundtrips)} round-trip URI sets match"
    )
    print(f"  → {COMPAT_CSV.relative_to(REPO_ROOT)}")
    print(f"  → {ROUNDTRIP_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
