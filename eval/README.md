# Evaluation Corpus & Experiments

One-line reproduction commands for each table/figure in §7 of the paper.

## Task 1 — Build the corpus

```bash
python -m eval.build_corpus               # synthetic half (offline)
python -m eval.harvest_real               # real half — needs internet
```

`build_corpus.py` writes the synthetic JSON-LD files and overwrites
`eval/corpus/manifest.csv` with synthetic rows. `harvest_real.py` is
**not** runnable in CI / sandbox (no internet); run it once from a
workstation and commit the contents of `eval/corpus/real/` plus the
appended manifest rows as a versioned snapshot.

| Output | Purpose |
|---|---|
| `eval/corpus/synthetic/*.jsonld` | 10 deterministic variants derived from the AgrifoodTEF maize fixture |
| `eval/corpus/real/*.jsonld` | 40 real records from data.europa.eu (committed snapshot) |
| `eval/corpus/manifest.csv` | single source of truth for N, n_real, n_syn |

## Task 2 — Vendor GeoDCAT-AP 3.0 SHACL shapes

```bash
pip install -r requirements-dev.txt           # pulls pyshacl + rdflib
python -m eval.vendor_shapes                  # one-time fetch — needs internet
```

`vendor_shapes.py` writes the SHACL shapes (GeoDCAT-AP 3.0 + DCAT-AP), the
imported vocabularies (DCAT, DCT, FOAF, ADMS), and the JSON-LD `@context`
files into `tests/fixtures/shacl/geodcat-ap-3.0/`, and rewrites the
provenance table in that folder's `PROVENANCE.md` with retrieval URLs,
sha256s, and timestamps. The fetch is not runnable in CI/sandbox — run it
on a workstation, commit the result, and the conformance gate runs
offline thereafter.

`tests/shacl_helpers.py` exposes `validate(graph) -> ValidationResult`
(session-cached shapes; `advanced=True` for SHACL-SPARQL; Violations vs
Warnings separated) and the `geodcat_ap_3_0_shapes` pytest fixture.

## Task 3 — E1 SHACL conformance (→ Table 1)

```bash
python -m eval.run_e1_conformance         # writes eval/results/e1_*.csv
pytest tests/test_geodcat_ap_conformance.py
```

Requires Task 1 corpus + Task 2 vendored shapes + dev deps installed.
Writes:
- `eval/results/e1_per_record.csv` — one row per corpus entry (conforms,
  n_violations, n_warnings, distinct warning shape IDs, first violation
  triage line)
- `eval/results/e1_conformance.csv` — Table 1: aggregated by
  `(output_type, origin)`
- `eval/results/e1_warning_taxonomy.csv` — Table 1 footnote: shape ID →
  count → representative human-readable cause

`tests/test_geodcat_ap_conformance.py` is the CI gate — parametrized over
manifest rows, fails on any `sh:Violation` in sidecar output. Warnings are
reported but do not fail the gate (the contribution claim is
Violation-clean, not Warning-clean). The test skips cleanly when shapes
aren't vendored yet, so CI in earlier states surfaces a setup error
instead of a false pass.

## Task 4 — E2 zero-loss augmentation (→ Table 2)

```bash
python -m eval.run_e2_zeroloss            # writes eval/results/e2_*.csv
pytest tests/test_zeroloss_augmentation.py
```

Requires Task 1 corpus + dev deps (rdflib). Runs offline against the
synthetic half on its own; processes real records too when present.
Writes:
- `eval/results/e2_per_record.csv` — one row per (record, strategy) with
  `client_triples_in`, `preserved`, `lost`, `preservation_rate`,
  `lost_predicates`, and a `custom_pass_through` flag for the
  `custom_predicates` synthetic row
- `eval/results/e2_zeroloss.csv` — Table 2: aggregated by
  `(origin, strategy)` for `augment` vs `regenerate_from_scratch`

`tests/test_zeroloss_augmentation.py` is the CI gate — for each AI-Ready
row, fails if the augmenter drops a client predicate outside the
explicit-replacement set (`dct:identifier`, `dct:publisher`,
`dcat:distribution`). Plus an explicit pass-through check on the four
`client.example/internal#` predicates the `custom_predicates` synthetic
row carries. Skips cleanly when rdflib isn't installed or the corpus
hasn't been built.

## Task 5 — E3 COCO compatibility + semantic round-trip (→ Table 3)

```bash
python -m eval.run_e3_coco                # writes eval/results/e3_*.csv
pytest tests/test_coco_compat.py
```

Requires the committed `tests/fixtures/sample-coco.json` plus the Task 1
corpus. Dev deps (`pycocotools`, `torchvision`) are required for the
full Table 3 — the runner still produces a CSV when they're missing, but
the rows for those loaders carry `parses_without_error = no` and the
import error in the diagnostic column. The `plain_json` and round-trip
columns work without CV deps.

Writes:
- `eval/results/e3_coco_compat.csv` — Table 3: one row per
  (sample, loader) with `parses_without_error`,
  `standard_fields_accessible`, `semantic_uri_preserved_or_ignored`,
  and a diagnostic column on failures
- `eval/results/e3_semantic_roundtrip.csv` — §7.4 round-trip:
  `theme_uris`, `semantic_uris`, `intersection`, `set_equal`, plus
  `missing_in_coco` / `extra_in_coco` for diff triage

Samples include the committed `sample-coco.json` (paired with the maize
input fixture) and one stub COCO per AI-Ready synthetic row whose input
carries AGROVOC themes. The committed fixture's COCO categories were
hand-curated against the input themes, so the strict
`semantic_uris == theme_uris` round-trip only holds for the
sidecar-generated synthetic pairs; the fixture's round-trip CSV row
reports the asymmetry rather than failing.

`tests/test_coco_compat.py` is the CI gate — `pycocotools` parses the
fixture, plain JSON preserves `semantic_uri`, and a sidecar-generated
synthetic pair achieves exact round-trip URI equality. `torchvision` is
exercised by the runner only (heavy import; skipped in the gate).
Skips cleanly when `pycocotools` isn't installed.

## Task 6 — E4 sidecar overhead (→ Figure 1)

```bash
# panel (b): augmentation cost vs input size (in-process, offline)
python -m eval.run_e4_perf

# panel (a): 202 latency under concurrency (sidecar must be running with
# stubbed Timeworx + staging — leave env vars unset so the sidecar picks
# the stubs). Repeat once per concurrency level the paper reports:
locust -f eval/locustfile.py --headless -u 1  -r 1  -t 60s \
    --host http://127.0.0.1:8000 --csv=eval/results/e4_latency_c1
locust -f eval/locustfile.py --headless -u 10 -r 10 -t 60s \
    --host http://127.0.0.1:8000 --csv=eval/results/e4_latency_c10
locust -f eval/locustfile.py --headless -u 50 -r 50 -t 60s \
    --host http://127.0.0.1:8000 --csv=eval/results/e4_latency_c50

# re-run run_e4_perf after locust to read its CSVs and draw the figure
python -m eval.run_e4_perf

pytest tests/test_broker_overhead.py
```

Requires the Task 1 corpus (uses `rich_full.jsonld` as the augmentation
base). Dev deps are optional for panel (b): the runner emits its CSV
with the standard library alone. matplotlib is needed to draw the
figure; locust is needed for panel (a).

Writes:
- `eval/results/e4_augment_cost.csv` — panel (b): `pad_predicates`,
  `input_triples_approx`, `n_runs`, `p50_ms`, `p95_ms`, `p99_ms`,
  `min_ms`, `max_ms`. Eight input sizes from 0 to 1600 pad predicates.
- `eval/results/e4_latency.csv` — panel (a): one row per concurrency
  level, produced by parsing locust's `*_stats.csv` outputs and tagging
  each by filename suffix
- `eval/results/fig1_broker_overhead.png` — combined Figure 1; drawn
  when matplotlib is installed and `e4_augment_cost.csv` exists. When
  panel (a) data is absent, panel (b) is drawn alone with a note.

`tests/test_broker_overhead.py` is the CI gate — asserts that
augmentation cost grows roughly linearly (the 4x-input ratio stays
under 12x, generous enough for CI jitter and tight enough to catch a
true O(n²) regression), and that the padded input survives the
augmenter end-to-end. Skips cleanly when the corpus hasn't been built.

Use a **stub segmentation backend** for panel (a) — leave
`TIMEWORX_API_BASE_URL`, `TIMEWORX_API_KEY`, `GCS_BUCKET` unset so
`dependencies.build_default_state` selects `StubTimeworxClient` +
`StubStagingService`. State this in the figure caption: the 202 latency
reflects sidecar overhead, not AI compute or storage I/O.

## Layout

```
eval/
├── build_corpus.py
├── harvest_real.py
├── run_e1_conformance.py
├── run_e2_zeroloss.py
├── run_e3_coco.py
├── run_e4_perf.py
├── locustfile.py
├── corpus/
│   ├── manifest.csv
│   ├── synthetic/
│   └── real/
└── results/                  (created by each runner)
```

Synthetic rows are regenerable; real rows and `eval/results/` are committed
as the reproducible record of a specific evaluation run.
