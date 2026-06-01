# GeoDCAT-AP 3.0 SHACL — Vendored Snapshot

This directory holds the official SEMIC GeoDCAT-AP 3.0 SHACL shapes plus the
imported vocabularies (DCAT, DCT, FOAF, ADMS) and the JSON-LD `@context`
files our test corpus references. All evaluation experiments
(`eval/run_e1_conformance.py`, etc.) read from this folder so a single
upstream commit pins the entire conformance gate.

> **Re-pinning.** Re-fetch with:
> ```bash
> python -m eval.vendor_shapes
> ```
> The script writes the table below in-place. The fetch needs internet — do
> it on a workstation, commit the changes, and the conformance gate runs
> offline thereafter.

## Pinned snapshot

| File | Upstream URL | SHA-256 | Retrieved (UTC) |
|---|---|---|---|
| shapes/geodcat-ap-shacl.ttl | `https://raw.githubusercontent.com/SEMICeu/GeoDCAT-AP/master/releases/3.0.0/shacl/geodcat-ap-SHACL.ttl` | `c1da0ebe3ad0f4be2a9b41a0b839142d6a31e2443f0f12650eaabd27cb71c39b` | 2026-05-31T22:55:24+00:00 |
| shapes/dcat-ap-shacl.ttl | `https://raw.githubusercontent.com/SEMICeu/DCAT-AP/master/releases/3.0.0/shacl/dcat-ap-SHACL.ttl` | `92f76609d78d257123e75bc6b7155df5cc0a63f14c29fbc12b0ac95c56af2059` | 2026-05-31T22:55:24+00:00 |
| vocab/dcat.ttl | `https://www.w3.org/ns/dcat.ttl` | `d1624d3bbf364f6fc1d43b9a188bbb9e67639b1a7d46a5cb7ea3f5d47d02b62f` | 2026-05-31T22:55:25+00:00 |
| vocab/dct.ttl | `https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_terms.ttl` | `13df401072dd7015bf9d75162f3e41c8138075304b7b9cc1aa1e9c16db976797` | 2026-05-31T22:55:25+00:00 |
| vocab/foaf.rdf | `http://xmlns.com/foaf/spec/index.rdf` | `eecbe127c9a3878f77df5f96dc77dce10ad7909b344be71e752b599386bc25d6` | 2026-05-31T22:55:25+00:00 |
| vocab/adms.ttl | `https://www.w3.org/ns/adms.ttl` | `634d8bfa4a9854c47cc4ebd1416362f6f05bf1bebb507d670a430ae838c1c8d2` | 2026-05-31T22:55:25+00:00 |
| contexts/geodcat-ap.jsonld | `https://raw.githubusercontent.com/SEMICeu/GeoDCAT-AP/master/releases/3.0.0/context/geodcat-ap.jsonld` | `d54be93f30a017db36789a809866523a08989b36ce65fa850575ce86c152e6c8` | 2026-05-31T22:55:26+00:00 |
| contexts/dcat3.jsonld | `https://www.w3.org/ns/dcat3.jsonld` | `48bbd53b0f3febb5e76bb1569868f29b815579006218b87f6f8bbfe9e196961d` | 2026-05-31T22:55:26+00:00 |
| contexts/dcat2.jsonld | `https://www.w3.org/ns/dcat2.jsonld` | `429babd00b95e5a8b2fefb4615a392988757526afec3ae78e6b50e94328462f4` | 2026-05-31T22:55:26+00:00 |

## Upstream sources

- **GeoDCAT-AP 3.0 shapes** — SEMICeu/GeoDCAT-AP repo, branch `master`,
  path `releases/3.0.0/`. Reference:
  <https://github.com/SEMICeu/GeoDCAT-AP/tree/master/releases/3.0.0>
- **DCAT-AP shapes** — imported transitively by GeoDCAT-AP; SEMICeu/DCAT-AP,
  `releases/3.0.0/`.
- **Vocabs (DCAT, DCT, FOAF, ADMS)** — canonical W3C / DCMI / FOAF / SEMIC
  URLs. The `vendor_shapes.py` script content-negotiates Turtle/RDF-XML.
- **JSON-LD contexts** — W3C `dcat3.jsonld`, `dcat2.jsonld` and the
  SEMICeu GeoDCAT-AP context.

## SHA-256 record

Computed at vendor-time over the file as written to disk. Verify with
`sha256sum <file>` matches the table above before treating the snapshot as
trusted.

## Why vendor

Two reasons:

1. **Reproducibility.** The paper's Table 1 numbers are computed against a
   specific shapes commit. If SEMICeu changes a shape between two runs the
   warning counts shift — the table becomes uninterpretable without a
   pinned version.
2. **Offline CI.** `pytest tests/test_geodcat_ap_conformance.py` must run
   without internet and without flakiness from upstream availability.

A `pyld` document loader override (see `tests/shacl_helpers.py`) maps the
upstream URLs back to the local files at parse time, so JSON-LD inputs
that reference the official context URLs validate against the vendored
copies transparently.
