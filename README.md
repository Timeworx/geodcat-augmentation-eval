# geodcat-augmentation-eval

Reproducible evaluation framework and corpus for measuring SHACL
conformance, zero-loss metadata augmentation, COCO toolchain
compatibility, and sidecar overhead in a GeoDCAT-AP 3.0 Data Space
augmentation pipeline. Backs Section 7 (Tables 1–3 + Figure 1) of the
accompanying paper (Bridging Semantic Web Standards and Computer Vision for AI-Ready Data in European Data Spaces)

## What's in here

```
src/metadata/      Augmentation library — augment_ai_ready,
                   build_legacy_skeleton, COCO semantic_uri injection,
                   GeoDCAT-AP vocab + JSON-LD helpers
eval/              Experiment runners + corpus builder + locustfile
eval/corpus/       60 records (10 synthetic + 50 real from data.europa.eu)
eval/results/      Pre-computed Table 1–3 + Figure 1 outputs
tests/             Four CI gates (one per experiment) + unit tests for
                   the augmentation library, plus the SHACL helpers and
                   the vendored GeoDCAT-AP 3.0 shapes under
                   tests/fixtures/shacl/
```

## TL;DR results

| Experiment | Result |
|---|---|
| **E1** — SHACL conformance | **59/60 (98.3%)** GeoDCAT-AP 3.0 conformant, 0 warnings |
| **E2** — Zero-loss augmentation | **86.1%** mean preservation vs **6.9–7.6%** for the regenerate-from-scratch baseline (~79pp delta) |
| **E3** — COCO toolchain compatibility | **28/28** (sample, loader) pairs parse + expose standard fields; `semantic_uri` rides through every loader that retains category extras |
| **E4** — Sidecar overhead | p99 = 4 ms at c=10 (228k requests, 0 failures); augmentation cost linear, 0.03–0.46 ms across 8 input sizes |

## Quick start

```bash
pip install -r requirements-dev.txt

# rebuild the corpus + vendor shapes (needs internet)
python -m eval.build_corpus
python -m eval.harvest_real            # 50 real records from data.europa.eu
python -m eval.vendor_shapes           # SHACL shapes + vocabs + contexts

# run all four experiments
python -m eval.run_e1_conformance      # → eval/results/e1_*.csv (Table 1)
python -m eval.run_e2_zeroloss         # → eval/results/e2_*.csv (Table 2)
python -m eval.run_e3_coco             # → eval/results/e3_*.csv (Table 3)
python -m eval.run_e4_perf             # → eval/results/e4_augment_cost.csv + fig1
# E4 panel (a) — 202 latency — needs a running HTTP service to measure;

# the four CI gates (one per experiment) plus unit tests
pytest tests/
```

The pre-computed `eval/results/` already contain the numbers that back
the paper's tables. Re-running the pipeline above regenerates them
deterministically; the synthetic corpus is regenerable and the real
corpus is a committed snapshot.

## What this repository does *not* contain

The full Data Space sidecar (FastAPI app, routers, GCS gateway, Timeworx
HTTP client, correlation store, dependency injection, auth seam,
deployment manifests) is intentionally not published. Only the
algorithmic contribution — the augmentation library — and the
evaluation framework around it are released. This keeps the
research artefact small and reusable without exposing the
productionisation plumbing.

## Licensing

- **Code** (`src/`, `eval/`, `tests/`): Apache License 2.0 — see `LICENSE`.
- **Data** (corpus, results, sample fixtures): CC-BY 4.0 — see
  `LICENSE-DATA`.
- **Vendored third-party material**: retains its upstream license. See
  `NOTICE` for attribution and license details for the SEMICeu SHACL
  shapes, W3C vocabularies, DCMI Terms, and FOAF.

## Citation

See `CITATION.cff` or the GitHub "Cite this repository" button. The
accompanying paper is the preferred citation for research use; cite the
Zenodo-archived release alongside it for reproducibility.

## Reproducibility note for paper reviewers

Every cell of every table in Section 7 of the "Bridging Semantic Web Standards and Computer Vision for AI-Ready Data in European Data Spaces" paper is regenerable from a
fresh clone of this repository via the commands in **Quick start**
above. Real corpus records and SHACL shapes are committed as
versioned snapshots so the experiment runs offline after the one-time
`harvest_real` and `vendor_shapes` fetches.
