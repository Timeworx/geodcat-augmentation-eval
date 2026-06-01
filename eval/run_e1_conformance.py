"""Experiment E1 — SHACL conformance against GeoDCAT-AP 3.0 (→ Table 1).

For each manifest row, route to the *real* augmenter or skeleton builder,
parse the JSON-LD output into rdflib, validate against the vendored shapes,
and record violations + warnings. Aggregate to Table 1.

Outputs:
    eval/results/e1_per_record.csv         one row per manifest entry
    eval/results/e1_conformance.csv        Table 1 — aggregated by (output_type, origin)
    eval/results/e1_warning_taxonomy.csv   Table 1 footnote — warning shape → count → cause

Run from the root folder (vendored shapes + dev deps required):

    python -m eval.run_e1_conformance
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from src.metadata.augmenter import augment_ai_ready
from src.metadata.skeleton import build_legacy_skeleton
from src.metadata.types import SegmentationType

from tests.shacl_helpers import ValidationFinding, ValidationResult, validate

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

PER_RECORD_CSV = RESULTS_DIR / "e1_per_record.csv"
TABLE1_CSV = RESULTS_DIR / "e1_conformance.csv"
WARNING_TAXONOMY_CSV = RESULTS_DIR / "e1_warning_taxonomy.csv"

# Placeholder COCO URL — E1 does not exercise the COCO surface, but the
# augmenter/skeleton both expect a value. Use a stable URL so the resulting
# graph is deterministic across runs.
_COCO_URL_TEMPLATE = "https://example.org/coco/{job_id}.json"

# E1 is shape-driven, not segmentation-driven. Fix one type so the outputs
# differ only by their inputs. INSTANCE matches the maize fixture.
_SEGMENTATION_TYPE = SegmentationType.INSTANCE


@dataclass
class ManifestRow:
    id: str
    path: str
    origin: str           # synthetic | real
    richness_tier: str    # minimal | partial | rich
    expected_path: str    # ai_ready | legacy
    known_input_issues: str = ""
    source_url: str = ""
    retrieved_at: str = ""


@dataclass
class PerRecord:
    id: str
    output_type: str          # ai_ready | legacy
    origin: str
    richness_tier: str
    conforms: bool
    n_violations: int
    n_warnings: int
    distinct_warning_shapes: tuple[str, ...] = field(default_factory=tuple)
    first_violation: str = ""  # short message for triage
    error: str = ""            # set if augment/skeleton/validate raised


# --- Manifest IO -----------------------------------------------------------


def load_manifest() -> list[ManifestRow]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not present — run `python -m eval.build_corpus` first."
        )
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return [ManifestRow(**row) for row in csv.DictReader(f)]


def load_input_dict(row: ManifestRow) -> dict:
    """Returns the corpus entry as a Python dict.

    AI-Ready: a JSON-LD `dict` with `@context` + `@graph`.
    Legacy: a flat `{rawDataUri, targetClass}` body.
    """
    path = REPO_ROOT / row.path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- Routing to the sidecar's real functions --------------------------------


def _augment_with_real(row: ManifestRow, body: dict) -> dict:
    return augment_ai_ready(
        body,
        job_id=row.id,
        segmentation_type=_SEGMENTATION_TYPE,
        coco_access_url=_COCO_URL_TEMPLATE.format(job_id=row.id),
    )


def _skeleton_with_real(row: ManifestRow, body: dict) -> dict:
    raw_uri = body.get("rawDataUri") or body.get("raw_uri") or ""
    target_class = body.get("targetClass") or body.get("target_class") or "object"
    return build_legacy_skeleton(
        job_id=row.id,
        segmentation_type=_SEGMENTATION_TYPE,
        raw_uri=raw_uri,
        target_class=target_class,
        coco_access_url=_COCO_URL_TEMPLATE.format(job_id=row.id),
    )


def build_output(row: ManifestRow) -> dict:
    body = load_input_dict(row)
    if row.expected_path == "ai_ready":
        return _augment_with_real(row, body)
    if row.expected_path == "legacy":
        return _skeleton_with_real(row, body)
    raise ValueError(f"row {row.id}: unknown expected_path={row.expected_path!r}")


# --- Validation + per-record ----------------------------------------------


def evaluate(row: ManifestRow) -> PerRecord:
    try:
        out = build_output(row)
    except Exception as e:  # noqa: BLE001 — surface as an evaluation failure
        return PerRecord(
            id=row.id,
            output_type=row.expected_path,
            origin=row.origin,
            richness_tier=row.richness_tier,
            conforms=False,
            n_violations=0,
            n_warnings=0,
            error=f"build_output: {type(e).__name__}: {e}",
        )
    try:
        result: ValidationResult = validate(out)
    except Exception as e:  # noqa: BLE001
        return PerRecord(
            id=row.id,
            output_type=row.expected_path,
            origin=row.origin,
            richness_tier=row.richness_tier,
            conforms=False,
            n_violations=0,
            n_warnings=0,
            error=f"validate: {type(e).__name__}: {e}",
        )
    distinct_warning_shapes = tuple(
        sorted({f.source_shape for f in result.warnings if f.source_shape})
    )
    first_violation = ""
    if result.violations:
        v = result.violations[0]
        first_violation = f"{v.source_shape} @ {v.focus_node}: {v.message[:160]}"
    return PerRecord(
        id=row.id,
        output_type=row.expected_path,
        origin=row.origin,
        richness_tier=row.richness_tier,
        conforms=result.conforms,
        n_violations=len(result.violations),
        n_warnings=len(result.warnings),
        distinct_warning_shapes=distinct_warning_shapes,
        first_violation=first_violation,
    )


# --- Aggregation -----------------------------------------------------------


_AGG_COLUMNS = [
    "output_type",
    "origin",
    "n_records",
    "n_conforming",
    "conformance_rate",
    "n_records_with_warnings",
    "total_violations",
    "total_warnings",
    "n_distinct_warning_shapes",
]


def aggregate(per_records: Iterable[PerRecord]) -> list[dict]:
    by_bucket: dict[tuple[str, str], list[PerRecord]] = {}
    for r in per_records:
        by_bucket.setdefault((r.output_type, r.origin), []).append(r)
    rows: list[dict] = []
    for (output_type, origin), group in sorted(by_bucket.items()):
        n = len(group)
        n_conf = sum(1 for r in group if r.conforms)
        n_with_warn = sum(1 for r in group if r.n_warnings > 0)
        total_v = sum(r.n_violations for r in group)
        total_w = sum(r.n_warnings for r in group)
        distinct_shapes = set()
        for r in group:
            distinct_shapes.update(r.distinct_warning_shapes)
        rows.append(
            {
                "output_type": output_type,
                "origin": origin,
                "n_records": n,
                "n_conforming": n_conf,
                "conformance_rate": f"{n_conf / n:.3f}" if n else "0.000",
                "n_records_with_warnings": n_with_warn,
                "total_violations": total_v,
                "total_warnings": total_w,
                "n_distinct_warning_shapes": len(distinct_shapes),
            }
        )
    return rows


def warning_taxonomy(per_records: Iterable[PerRecord]) -> list[dict]:
    """Shape ID → count → most-frequent human-readable message.

    The Table 1 footnote needs a short story per recurring shape. We pick
    the *most common* `sh:resultMessage` for each shape ID as the
    representative cause — the rest of the variance tends to be IRI noise.
    """
    counter: Counter[str] = Counter()
    msg_buckets: dict[str, Counter[str]] = {}
    for r in per_records:
        # The per-record tuple only carries shape IDs, not messages. Re-walk
        # the live findings: aggregate() is called *after* evaluate(), so
        # the message data is lost. We collect message-by-shape inside the
        # evaluate loop via a side dict instead — see run().
        pass
    return []  # populated in run() via the live findings


# --- Main loop -------------------------------------------------------------


def _write_per_record(rows: list[PerRecord]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "output_type",
        "origin",
        "richness_tier",
        "conforms",
        "n_violations",
        "n_warnings",
        "distinct_warning_shapes",
        "first_violation",
        "error",
    ]
    with PER_RECORD_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r.id,
                    "output_type": r.output_type,
                    "origin": r.origin,
                    "richness_tier": r.richness_tier,
                    "conforms": "yes" if r.conforms else "no",
                    "n_violations": r.n_violations,
                    "n_warnings": r.n_warnings,
                    "distinct_warning_shapes": "; ".join(r.distinct_warning_shapes),
                    "first_violation": r.first_violation,
                    "error": r.error,
                }
            )


def _write_table1(agg_rows: list[dict]) -> None:
    with TABLE1_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_COLUMNS)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)


def _write_warning_taxonomy(taxonomy: list[dict]) -> None:
    cols = ["shape_id", "count", "representative_message"]
    with WARNING_TAXONOMY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in taxonomy:
            w.writerow(row)


def run() -> tuple[list[PerRecord], list[dict], list[dict]]:
    """Drive E1 end-to-end.

    Re-walks `validate(...)`'s output once per record so the warning
    taxonomy can capture messages (the `PerRecord` dataclass only retains
    shape IDs to keep the CSV small).
    """
    manifest = load_manifest()
    per_records: list[PerRecord] = []
    shape_counter: Counter[str] = Counter()
    shape_msg_counter: dict[str, Counter[str]] = {}

    for row in manifest:
        # Build the augmented/skeleton output once, then both record + count.
        try:
            out = build_output(row)
            result = validate(out)
        except Exception as e:  # noqa: BLE001
            per_records.append(
                PerRecord(
                    id=row.id,
                    output_type=row.expected_path,
                    origin=row.origin,
                    richness_tier=row.richness_tier,
                    conforms=False,
                    n_violations=0,
                    n_warnings=0,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            continue

        distinct = tuple(
            sorted({f.source_shape for f in result.warnings if f.source_shape})
        )
        first_v = ""
        if result.violations:
            v = result.violations[0]
            first_v = f"{v.source_shape} @ {v.focus_node}: {v.message[:160]}"

        per_records.append(
            PerRecord(
                id=row.id,
                output_type=row.expected_path,
                origin=row.origin,
                richness_tier=row.richness_tier,
                conforms=result.conforms,
                n_violations=len(result.violations),
                n_warnings=len(result.warnings),
                distinct_warning_shapes=distinct,
                first_violation=first_v,
            )
        )

        # Build warning taxonomy live so we keep the messages.
        for f in result.warnings:
            sid = f.source_shape or "(anonymous)"
            shape_counter[sid] += 1
            shape_msg_counter.setdefault(sid, Counter())[f.message] += 1

    taxonomy_rows = [
        {
            "shape_id": sid,
            "count": count,
            "representative_message": _most_common(shape_msg_counter.get(sid)),
        }
        for sid, count in shape_counter.most_common()
    ]
    agg_rows = aggregate(per_records)
    return per_records, agg_rows, taxonomy_rows


def _most_common(c: Optional[Counter[str]]) -> str:
    if not c:
        return ""
    return c.most_common(1)[0][0][:240]


def main() -> None:
    per_records, agg_rows, taxonomy = run()
    _write_per_record(per_records)
    _write_table1(agg_rows)
    _write_warning_taxonomy(taxonomy)
    n = len(per_records)
    n_conf = sum(1 for r in per_records if r.conforms)
    n_err = sum(1 for r in per_records if r.error)
    print(
        f"E1: {n_conf}/{n} conformant ({n - n_conf - n_err} non-conforming, "
        f"{n_err} errored)"
    )
    print(f"  → {PER_RECORD_CSV.relative_to(REPO_ROOT)}")
    print(f"  → {TABLE1_CSV.relative_to(REPO_ROOT)}")
    print(f"  → {WARNING_TAXONOMY_CSV.relative_to(REPO_ROOT)}")
    if n_err or any(not r.conforms and not r.error for r in per_records):
        # Non-conformance is the experimental finding to report, not a fatal
        # condition. Errors are.
        if n_err:
            print(f"  {n_err} record(s) errored — see `error` column", file=sys.stderr)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
