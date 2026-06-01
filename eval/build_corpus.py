"""Generate the synthetic half of the evaluation corpus (Task 1.1).

The richest case (`rich_full.jsonld`) is the AgrifoodTEF maize record vendored
under `tests/fixtures/input-geodcat.jsonld`. Every other synthetic variant is
*derived* from that base via dict transforms — never hand-maintain a near-
duplicate JSON file.

Outputs:
    eval/corpus/synthetic/rich_full.jsonld
    eval/corpus/synthetic/minimal.jsonld
    eval/corpus/synthetic/missing_publisher.jsonld
    eval/corpus/synthetic/missing_spatial.jsonld
    eval/corpus/synthetic/missing_temporal.jsonld
    eval/corpus/synthetic/multi_geometry.jsonld
    eval/corpus/synthetic/custom_predicates.jsonld
    eval/corpus/synthetic/legacy_raw_01.jsonld
    eval/corpus/synthetic/legacy_raw_02.jsonld
    eval/corpus/synthetic/legacy_raw_03.jsonld
    eval/corpus/manifest.csv   (writes synthetic rows; harvest_real.py appends)

Run from the root:
    python -m eval.build_corpus
"""
from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "input-geodcat.jsonld"
SYN_DIR = REPO_ROOT / "eval" / "corpus" / "synthetic"
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"

CUSTOM_NS_PREFIX = "client_internal"
CUSTOM_NS_IRI = "https://client.example/internal#"

MANIFEST_COLUMNS = [
    "id",
    "path",
    "origin",
    "richness_tier",
    "expected_path",
    "known_input_issues",
    "source_url",
    "retrieved_at",
]


@dataclass
class SyntheticEntry:
    """One synthetic corpus row.

    `transform` mutates a deep-cloned base graph in place; for the Legacy
    variants the transform ignores the base and returns the raw descriptor.
    """

    id: str
    transform: Callable[[dict], Optional[dict]]
    richness_tier: str  # minimal | partial | rich
    expected_path: str  # ai_ready | legacy
    known_input_issues: str = ""


# --- Transforms ------------------------------------------------------------


def _dataset_node(graph: dict) -> dict:
    """Return the @graph[0] dataset node (the maize fixture has one)."""
    return graph["@graph"][0]


def _identity(graph: dict) -> None:
    pass


def _to_minimal(graph: dict) -> None:
    dataset = _dataset_node(graph)
    minimal_node = {"@id": dataset["@id"], "@type": "dcat:Dataset"}
    graph["@graph"] = [minimal_node]
    graph["@context"] = {"dcat": "http://www.w3.org/ns/dcat#"}


def _drop_predicate(predicate: str):
    def t(graph: dict) -> None:
        dataset = _dataset_node(graph)
        dataset.pop(predicate, None)
    return t


def _add_second_geometry(graph: dict) -> None:
    """Add a GeoJSON-shaped `locn:geometry` alongside the existing
    `dcat:bbox` WKT. GeoDCAT-AP 3.0 SHACL flags "multiple geometry encodings
    on the same Location" — this is the variant E1 will catch as a Warning.
    """
    dataset = _dataset_node(graph)
    spatial = dataset.get("dct:spatial")
    if not isinstance(spatial, dict):
        raise RuntimeError("multi_geometry variant requires dict-shaped dct:spatial")
    graph["@context"].setdefault("locn", "http://www.w3.org/ns/locn#")
    spatial["locn:geometry"] = {
        "@type": "geo:wktLiteral",
        "@value": (
            "POLYGON((15.12 48.12, 15.15 48.12, 15.15 48.15, "
            "15.12 48.15, 15.12 48.12))"
        ),
    }
    # And a GeoJSON-typed alternate, as some publishers emit both forms.
    spatial["dcat:centroid"] = {
        "@type": "geo:wktLiteral",
        "@value": "POINT(15.135 48.135)",
    }


def _add_custom_predicates(graph: dict) -> None:
    """Attach four predicates outside the standard vocabularies. E2 asserts
    these survive end-to-end in the augmented output (zero-loss)."""
    dataset = _dataset_node(graph)
    graph["@context"].setdefault(CUSTOM_NS_PREFIX, CUSTOM_NS_IRI)
    dataset[f"{CUSTOM_NS_PREFIX}:sensorSerial"] = "UAV-2025-WB-07"
    dataset[f"{CUSTOM_NS_PREFIX}:campaignCode"] = "WIE-MAIZE-2025-08"
    dataset[f"{CUSTOM_NS_PREFIX}:droneAltitudeM"] = {
        "@type": "http://www.w3.org/2001/XMLSchema#decimal",
        "@value": "60.0",
    }
    dataset[f"{CUSTOM_NS_PREFIX}:pilotOperator"] = {
        "@id": "https://client.example/people/aria-novak",
        "foaf:name": "Aria Novak",
    }


# --- Legacy raw descriptors ------------------------------------------------
# Not GeoDCAT-AP graphs. Route via `map_legacy` (rawDataUri + targetClass).
# Variant by source-URI shape, since the sidecar's source-resolver dispatch
# and the catalog skeleton both key off the URI scheme.

LEGACY_VARIANTS: list[dict] = [
    {
        "rawDataUri": "s3://agrifoodtef-client-bucket/raw/wieselburg-2025/batch-01/",
        "targetClass": "maize plant",
    },
    {
        "rawDataUri": "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456/view?usp=sharing",
        "targetClass": "cauliflower",
    },
    {
        "rawDataUri": "https://storage.googleapis.com/agrifoodtef-public/datasets/grapevine-2025.zip",
        "targetClass": "grapevine",
    },
]


def _legacy_factory(index: int) -> Callable[[dict], dict]:
    """Returns a transform that ignores the cloned base and emits a legacy
    descriptor instead. The returned dict replaces the cloned graph."""

    def t(_graph: dict) -> dict:
        return dict(LEGACY_VARIANTS[index])

    return t


# --- Catalog ---------------------------------------------------------------


SYNTHETIC: list[SyntheticEntry] = [
    SyntheticEntry(
        id="rich_full",
        transform=_identity,
        richness_tier="rich",
        expected_path="ai_ready",
        known_input_issues="",
    ),
    SyntheticEntry(
        id="minimal",
        transform=_to_minimal,
        richness_tier="minimal",
        expected_path="ai_ready",
        known_input_issues="missing dct:title; missing dct:publisher; missing dct:description; missing dcat:distribution",
    ),
    SyntheticEntry(
        id="missing_publisher",
        transform=_drop_predicate("dct:publisher"),
        richness_tier="partial",
        expected_path="ai_ready",
        known_input_issues="missing dct:publisher",
    ),
    SyntheticEntry(
        id="missing_spatial",
        transform=_drop_predicate("dct:spatial"),
        richness_tier="partial",
        expected_path="ai_ready",
        known_input_issues="missing dct:spatial",
    ),
    SyntheticEntry(
        id="missing_temporal",
        transform=_drop_predicate("dct:temporal"),
        richness_tier="partial",
        expected_path="ai_ready",
        known_input_issues="missing dct:temporal",
    ),
    SyntheticEntry(
        id="multi_geometry",
        transform=_add_second_geometry,
        richness_tier="rich",
        expected_path="ai_ready",
        known_input_issues="multiple geometry encodings on dct:spatial (GeoDCAT-AP 3.0 SHACL warning)",
    ),
    SyntheticEntry(
        id="custom_predicates",
        transform=_add_custom_predicates,
        richness_tier="rich",
        expected_path="ai_ready",
        known_input_issues="non-standard predicates under https://client.example/internal# (zero-loss test)",
    ),
    SyntheticEntry(
        id="legacy_raw_01",
        transform=_legacy_factory(0),
        richness_tier="minimal",
        expected_path="legacy",
        known_input_issues="raw S3 URI; no metadata",
    ),
    SyntheticEntry(
        id="legacy_raw_02",
        transform=_legacy_factory(1),
        richness_tier="minimal",
        expected_path="legacy",
        known_input_issues="raw Google Drive share URL; no metadata",
    ),
    SyntheticEntry(
        id="legacy_raw_03",
        transform=_legacy_factory(2),
        richness_tier="minimal",
        expected_path="legacy",
        known_input_issues="raw HTTPS zip URL; no metadata",
    ),
]


# --- Build -----------------------------------------------------------------


def _load_base() -> dict:
    with BASE_FIXTURE.open("r", encoding="utf-8") as f:
        return json.load(f)


def build() -> list[dict]:
    """Generate all synthetic files and return manifest rows."""
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base()
    rows: list[dict] = []
    for entry in SYNTHETIC:
        cloned = copy.deepcopy(base)
        result = entry.transform(cloned)
        out = result if isinstance(result, dict) and entry.expected_path == "legacy" else cloned
        path = SYN_DIR / f"{entry.id}.jsonld"
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        rows.append(
            {
                "id": entry.id,
                "path": str(path.relative_to(REPO_ROOT)),
                "origin": "synthetic",
                "richness_tier": entry.richness_tier,
                "expected_path": entry.expected_path,
                "known_input_issues": entry.known_input_issues,
                # Synthetic provenance: GeoDCAT variants derive from the maize
                # fixture; Legacy raws are invented descriptors, no source.
                "source_url": (
                    "tests/fixtures/input-geodcat.jsonld"
                    if entry.expected_path == "ai_ready"
                    else ""
                ),
                "retrieved_at": "",
            }
        )
    return rows


def write_manifest_synthetic(rows: list[dict]) -> None:
    """Overwrite the manifest with synthetic rows.

    `harvest_real.py` appends real rows in-place after this runs. Rerunning
    `build_corpus.py` resets to synthetic-only — the harvester is idempotent
    against its own dedupe key, so re-appending after a rebuild is safe.
    """
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    rows = build()
    write_manifest_synthetic(rows)
    print(f"wrote {len(rows)} synthetic entries -> {SYN_DIR}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
