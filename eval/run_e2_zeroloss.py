"""Experiment E2 — zero-loss augmentation (→ Table 2).

For each AI-Ready manifest row, run the *real* `augment_ai_ready` and a
baseline `regenerate_from_scratch` strategy, then diff each output against
the input on the dataset node's (predicate, object) triple set. The
delta between the two strategies is the contribution's payoff.

Preservation accounting (per the brief):
  * Direct: same (predicate, object) on the output dataset node.
  * Through prov:wasDerivedFrom: covers the augmenter's intentional
    re-rooting of @id and the carry-over of dct:title (and
    dct:description as the wasDerivedFrom node's dct:type).
  * Non-Literal objects (publisher, spatial, temporal, distributions)
    compare by structural subtree, not by blank-node identity — bnodes
    get fresh internal IDs on each rdflib parse.

Outputs:
    eval/results/e2_per_record.csv    one row per (record, strategy)
    eval/results/e2_zeroloss.csv      Table 2 — aggregated by
                                      (origin, strategy)

Run from the root folder:

    python -m eval.run_e2_zeroloss

Prereqs: dev deps (rdflib) installed; `python -m eval.build_corpus` ran.
Synthetic corpus alone is sufficient — real records are diffed too when
present.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.metadata.augmenter import augment_ai_ready
from src.metadata.jsonld import find_dataset
from src.metadata.types import SegmentationType
from src.metadata import vocab

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

PER_RECORD_CSV = RESULTS_DIR / "e2_per_record.csv"
SUMMARY_CSV = RESULTS_DIR / "e2_zeroloss.csv"

_COCO_URL_TEMPLATE = "https://example.org/coco/{job_id}.json"
_SEGMENTATION_TYPE = SegmentationType.INSTANCE

_DCT = "http://purl.org/dc/terms/"
_DCAT = "http://www.w3.org/ns/dcat#"
_PROV = "http://www.w3.org/ns/prov#"

_DCT_DESCRIPTION = _DCT + "description"
_DCT_TYPE = _DCT + "type"
_PROV_WAS_DERIVED_FROM = _PROV + "wasDerivedFrom"

# The synthetic `custom_predicates` row attaches these four predicates
# outside the standard vocabularies. The augmenter must pass them through
# untouched — explicit assertion (not just preservation_rate).
_CUSTOM_PREDS = (
    "https://client.example/internal#sensorSerial",
    "https://client.example/internal#campaignCode",
    "https://client.example/internal#droneAltitudeM",
    "https://client.example/internal#pilotOperator",
)


@dataclass
class PerRecord:
    id: str
    origin: str
    richness_tier: str
    strategy: str               # augment | regenerate_from_scratch
    client_triples_in: int
    preserved: int
    lost: int
    preservation_rate: float
    lost_predicates: str        # ";"-joined distinct predicate IRIs
    custom_pass_through: str    # n/a | yes | no (N missing)
    error: str = ""


# --- Skip handling --------------------------------------------------------


def _missing_rdflib_reason() -> str | None:
    try:
        import rdflib  # noqa: F401
    except ImportError as e:
        return (
            f"rdflib missing ({e.name}); "
            f"run `pip install -r requirements-dev.txt`"
        )
    return None


def _missing_manifest_reason() -> str | None:
    if not MANIFEST.exists():
        return (
            f"manifest missing at {MANIFEST}; "
            f"run `python -m eval.build_corpus` first"
        )
    return None


# --- Graph helpers --------------------------------------------------------


def _to_graph(doc: dict):
    """Parse a JSON-LD dict to an rdflib.Graph. Installs the offline pyld
    loader when the vendored contexts are present so real records (which
    carry URL-shaped @contexts) resolve to local files; synthetic records
    use inline contexts and don't need it.
    """
    from rdflib import Graph
    try:
        from tests.shacl_helpers import CONTEXTS_DIR, install_offline_context_loader
        if CONTEXTS_DIR.exists() and any(CONTEXTS_DIR.iterdir()):
            install_offline_context_loader()
    except Exception:  # noqa: BLE001 — best-effort, synthetic still parses
        pass
    g = Graph()
    g.parse(data=json.dumps(doc), format="json-ld")
    return g


def _find_dataset_uri(g):
    from rdflib import RDF, URIRef
    DCAT_DATASET = URIRef(_DCAT + "Dataset")
    for s in g.subjects(RDF.type, DCAT_DATASET):
        return s
    return None


def _dataset_pred_objs(g, root) -> list[tuple]:
    """Return (pred, obj) on the dataset node, excluding the rdf:type edge."""
    from rdflib import RDF
    return [(p, o) for p, o in g.predicate_objects(root) if p != RDF.type]


def _literal_signature(lit) -> str:
    return f"{lit.value!r}|{lit.language or ''}|{lit.datatype or ''}"


def _normalize_subtree(g, root) -> frozenset:
    """Canonicalize a non-literal object as a frozenset of
    (predicate_path, leaf_descriptor) pairs. Lets us compare structured
    objects (publisher, spatial, distribution) by content rather than by
    blank-node identity — rdflib mints fresh BNode IDs on each parse.
    """
    from rdflib.term import BNode, Literal, URIRef

    leaves: set[tuple] = set()
    seen_ids: set[int] = set()

    def walk(node, path):
        if id(node) in seen_ids:
            leaves.add((path + ("<cycle>",), str(node)))
            return
        seen_ids.add(id(node))
        if isinstance(node, URIRef):
            leaves.add((path + ("@id",), str(node)))
        for p, o in g.predicate_objects(node):
            new_path = path + (str(p),)
            if isinstance(o, Literal):
                leaves.add((new_path, _literal_signature(o)))
            else:
                walk(o, new_path)

    walk(root, ())
    return frozenset(leaves)


def _matches(input_obj, candidate_obj, in_g, out_g) -> bool:
    from rdflib.term import BNode, Literal, URIRef
    if isinstance(input_obj, Literal):
        if isinstance(candidate_obj, Literal):
            return _literal_signature(candidate_obj) == _literal_signature(input_obj)
        return False
    if isinstance(input_obj, URIRef):
        if isinstance(candidate_obj, URIRef):
            return str(candidate_obj) == str(input_obj)
        return False
    if isinstance(input_obj, BNode):
        if isinstance(candidate_obj, (BNode, URIRef)):
            return _normalize_subtree(in_g, input_obj) == _normalize_subtree(out_g, candidate_obj)
    return False


def _wd_objects(g, dataset_uri) -> list:
    from rdflib import URIRef
    return list(g.objects(dataset_uri, URIRef(_PROV_WAS_DERIVED_FROM)))


def _preserved(input_pred, input_obj, in_g, out_g, out_root) -> bool:
    """Is (input_pred, input_obj) preserved somewhere in the output?"""
    # 1. Direct on the output dataset node.
    for p, o in out_g.predicate_objects(out_root):
        if p == input_pred and _matches(input_obj, o, in_g, out_g):
            return True

    # 2. Under any prov:wasDerivedFrom (catches the augmenter's
    #    title/description re-rooting).
    for wd in _wd_objects(out_g, out_root):
        for p, o in out_g.predicate_objects(wd):
            if p == input_pred and _matches(input_obj, o, in_g, out_g):
                return True
            # 2a. dct:description on input → dct:type on the wasDerivedFrom
            #     node (an intentional remap inside the augmenter).
            if (
                str(input_pred) == _DCT_DESCRIPTION
                and str(p) == _DCT_TYPE
                and _matches(input_obj, o, in_g, out_g)
            ):
                return True

    return False


# --- Strategy A: real augmenter -------------------------------------------


def _run_augment(input_doc: dict, job_id: str) -> dict:
    return augment_ai_ready(
        input_doc,
        job_id=job_id,
        segmentation_type=_SEGMENTATION_TYPE,
        coco_access_url=_COCO_URL_TEMPLATE.format(job_id=job_id),
    )


# --- Strategy B: regenerate-from-scratch baseline -------------------------


def _run_regenerate_from_scratch(input_doc: dict, job_id: str) -> dict:
    """Re-emit a fresh GeoDCAT-AP record from only the fields the sidecar
    explicitly extracts: title, description, publisher (Timeworx),
    distribution (COCO), processing keywords. Mirrors what
    build_legacy_skeleton does for the Legacy path, applied to AI-Ready
    inputs — everything outside that narrow schema (spatial, temporal,
    themes, client keywords, custom predicates) is lost. This is the
    foil; the delta vs. `augment` is Table 2's headline.
    """
    dataset = find_dataset(input_doc)
    original_title = dataset.get("dct:title")
    original_description = dataset.get("dct:description")
    dataset_iri = f"{vocab.TIMEWORX_DATASET_NS}/{job_id}"
    seg_pretty = _SEGMENTATION_TYPE.pretty
    title_out = f"{original_title or 'Untitled Dataset'} — {seg_pretty} Segmentation"
    coco_url = _COCO_URL_TEMPLATE.format(job_id=job_id)
    return {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "prov": "http://www.w3.org/ns/prov#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        },
        "@graph": [
            {
                "@id": dataset_iri,
                "@type": "dcat:Dataset",
                "dct:identifier": job_id,
                "dct:title": title_out,
                "dct:description": original_description or "",
                "dct:publisher": vocab.timeworx_publisher(),
                "dcat:keyword": [
                    {"@language": "en", "@value": kw}
                    for kw in vocab.PROCESSING_KEYWORDS[_SEGMENTATION_TYPE]
                ],
                "dcat:distribution": [vocab.coco_distribution(dataset_iri, coco_url)],
            }
        ],
    }


# --- Per-record diff ------------------------------------------------------


def _diff(input_doc: dict, output_doc: dict, *, row_id: str) -> tuple[int, int, list[str]]:
    in_g = _to_graph(input_doc)
    out_g = _to_graph(output_doc)
    in_root = _find_dataset_uri(in_g)
    out_root = _find_dataset_uri(out_g)
    if in_root is None:
        raise RuntimeError(f"no dcat:Dataset in input of {row_id}")
    if out_root is None:
        raise RuntimeError(f"no dcat:Dataset in output of {row_id}")
    pred_objs = _dataset_pred_objs(in_g, in_root)
    preserved = 0
    lost_preds: list[str] = []
    for p, o in pred_objs:
        if _preserved(p, o, in_g, out_g, out_root):
            preserved += 1
        else:
            lost_preds.append(str(p))
    return len(pred_objs), preserved, lost_preds


def _custom_pass_through(output_doc: dict) -> str:
    """Confirm the four custom_predicates synthetic-row predicates ride
    through end-to-end. Returns 'yes', 'no (N missing: …)', or 'n/a'.
    """
    out_g = _to_graph(output_doc)
    root = _find_dataset_uri(out_g)
    if root is None:
        return "n/a"
    from rdflib import URIRef
    missing = []
    for pred in _CUSTOM_PREDS:
        if next(out_g.objects(root, URIRef(pred)), None) is None:
            missing.append(pred.rsplit("#", 1)[-1])
    if not missing:
        return "yes"
    return f"no ({len(missing)} missing: {','.join(missing)})"


# --- Main loop ------------------------------------------------------------


def _load_manifest() -> list[dict]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ai_ready_rows(manifest: list[dict]) -> Iterable[dict]:
    for row in manifest:
        if row.get("expected_path") == "ai_ready":
            yield row


def _eval_one(input_doc: dict, row: dict, *, strategy: str) -> PerRecord:
    rid = row["id"]
    base = dict(
        id=rid, origin=row["origin"], richness_tier=row["richness_tier"],
        strategy=strategy, client_triples_in=0, preserved=0, lost=0,
        preservation_rate=0.0, lost_predicates="", custom_pass_through="n/a",
    )
    try:
        if strategy == "augment":
            output = _run_augment(input_doc, rid)
        elif strategy == "regenerate_from_scratch":
            output = _run_regenerate_from_scratch(input_doc, rid)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        client_in, preserved, lost_preds = _diff(input_doc, output, row_id=rid)
    except Exception as e:  # noqa: BLE001 — surface as evaluation error
        return PerRecord(**base, error=f"{type(e).__name__}: {e}")
    rate = preserved / client_in if client_in else 1.0
    custom = _custom_pass_through(output) if rid == "custom_predicates" else "n/a"
    return PerRecord(
        id=rid, origin=row["origin"], richness_tier=row["richness_tier"],
        strategy=strategy, client_triples_in=client_in, preserved=preserved,
        lost=client_in - preserved, preservation_rate=round(rate, 4),
        lost_predicates=";".join(sorted(set(lost_preds))),
        custom_pass_through=custom,
    )


def run() -> list[PerRecord]:
    manifest = _load_manifest()
    out: list[PerRecord] = []
    for row in _ai_ready_rows(manifest):
        input_doc = json.loads((REPO_ROOT / row["path"]).read_text(encoding="utf-8"))
        out.append(_eval_one(input_doc, row, strategy="augment"))
        out.append(_eval_one(input_doc, row, strategy="regenerate_from_scratch"))
    return out


# --- CSV writers ----------------------------------------------------------


def _write_per_record(rows: list[PerRecord]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else [
        f.name for f in PerRecord.__dataclass_fields__.values()
    ]
    with PER_RECORD_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


_SUMMARY_COLUMNS = [
    "origin",
    "strategy",
    "n_records",
    "client_triples_in_total",
    "preserved_total",
    "lost_total",
    "mean_preservation_rate",
    "min_preservation_rate",
    "max_preservation_rate",
]


def _write_summary(rows: list[PerRecord]) -> None:
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "n": 0, "in": 0, "kept": 0, "min": 1.0, "max": 0.0,
    })
    for r in rows:
        if r.error:
            continue
        key = (r.origin, r.strategy)
        a = agg[key]
        a["n"] += 1
        a["in"] += r.client_triples_in
        a["kept"] += r.preserved
        a["min"] = min(a["min"], r.preservation_rate)
        a["max"] = max(a["max"], r.preservation_rate)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        w.writeheader()
        for (origin, strategy), a in sorted(agg.items()):
            mean = a["kept"] / a["in"] if a["in"] else 1.0
            w.writerow({
                "origin": origin,
                "strategy": strategy,
                "n_records": a["n"],
                "client_triples_in_total": a["in"],
                "preserved_total": a["kept"],
                "lost_total": a["in"] - a["kept"],
                "mean_preservation_rate": f"{mean:.4f}",
                "min_preservation_rate": f"{a['min']:.4f}",
                "max_preservation_rate": f"{a['max']:.4f}",
            })


def main() -> None:
    skip = _missing_rdflib_reason() or _missing_manifest_reason()
    if skip:
        print(f"E2 skipped: {skip}", file=sys.stderr)
        raise SystemExit(0)

    rows = run()
    _write_per_record(rows)
    _write_summary(rows)

    n_aug = sum(1 for r in rows if r.strategy == "augment" and not r.error)
    n_aug_lossy = sum(
        1 for r in rows
        if r.strategy == "augment" and not r.error and r.lost > 0
    )
    n_err = sum(1 for r in rows if r.error)
    print(
        f"E2: {n_aug} augment runs ({n_aug - n_aug_lossy} zero-loss, "
        f"{n_aug_lossy} with intentional replacements counted as lost), "
        f"{n_err} errored"
    )
    print(f"  → {PER_RECORD_CSV.relative_to(REPO_ROOT)}")
    print(f"  → {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    if n_err:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
