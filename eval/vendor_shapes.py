"""Vendor the GeoDCAT-AP 3.0 SHACL shapes + imports + JSON-LD contexts
(Task 2 of docs/Timeworx_eval_corpus_and_experiments_brief.md).

Run from the root folder, once, on a workstation with internet:

    python -m eval.vendor_shapes

Outputs into `tests/fixtures/shacl/geodcat-ap-3.0/`:

    shapes/geodcat-ap-shacl.ttl     # the GeoDCAT-AP 3.0 SHACL shapes
    shapes/dcat-ap-shacl.ttl        # imported by GeoDCAT-AP
    vocab/dcat.ttl                  # W3C DCAT vocab
    vocab/dct.ttl                   # DCMI Terms
    vocab/foaf.rdf                  # FOAF (RDF/XML; FOAF doesn't serve TTL)
    vocab/adms.ttl                  # ADMS vocab
    contexts/geodcat-ap.jsonld      # SEMICeu GeoDCAT-AP context
    contexts/dcat3.jsonld           # W3C DCAT 3 context
    contexts/dcat2.jsonld           # W3C DCAT 2 context

Then rewrites the table in `PROVENANCE.md` with the upstream URL, sha256,
and retrieval timestamp for every file. Re-running overwrites all entries
(use a pinned commit in `_SOURCES` to keep snapshots stable across reruns).

Why this is a script rather than a one-shot download in CI:

  * data.europa.eu, the SEMICeu repo, and several W3C namespace docs are
    not on the sandbox allow-list, and we explicitly do not want CI to
    depend on internet availability for the SHACL conformance gate.
  * Tying the snapshot to a specific upstream commit (see `_SHAPES_COMMIT`)
    pins Table 1's numbers — re-bumping the commit re-bumps the table.
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "tests" / "fixtures" / "shacl" / "geodcat-ap-3.0"
PROVENANCE = VENDOR_DIR / "PROVENANCE.md"

# Pinned upstream commits.
#
# Bump these together when re-vendoring. The shapes commit is the load-
# bearing one — change it deliberately, with a paper-table re-run.
_SHAPES_COMMIT = "master"      # SEMICeu publishes 3.0.0 on master; lock to commit if needed.
_DCAT_AP_COMMIT = "master"

USER_AGENT = "timeworx-sidecar-eval/0.1 (+https://timeworx.io)"
HTTP_TIMEOUT_S = 60.0


@dataclass
class VendorEntry:
    relpath: str
    url: str
    # Fallback URLs tried in order if `url` 404s. Lets us cope with
    # casing changes / folder reorgs in upstream without a code-edit
    # cycle.
    alt_urls: tuple[str, ...] = ()


_GEODCAT_BASE = (
    f"https://raw.githubusercontent.com/SEMICeu/GeoDCAT-AP/{_SHAPES_COMMIT}"
)


_SOURCES: list[VendorEntry] = [
    # SHACL shapes ----------------------------------------------------------
    VendorEntry(
        relpath="shapes/geodcat-ap-shacl.ttl",
        # Try the upper-case SHACL spelling first — the DCAT-AP sibling
        # uses `SHACL.ttl` (upper-case) at the same path shape, and the
        # lower-case variant 404s on the current commit. Lower-case is
        # kept as a fallback in case a future re-pin flips it back.
        url=f"{_GEODCAT_BASE}/releases/3.0.0/shacl/geodcat-ap-SHACL.ttl",
        alt_urls=(
            f"{_GEODCAT_BASE}/releases/3.0.0/shacl/geodcat-ap-shacl.ttl",
            f"{_GEODCAT_BASE}/releases/3.0.0/shacl/geodcat-ap.shapes.ttl",
            f"{_GEODCAT_BASE}/releases/3.0.0/shacl/geodcat-ap.shacl.ttl",
            f"{_GEODCAT_BASE}/shacl/geodcat-ap-SHACL.ttl",
        ),
    ),
    VendorEntry(
        relpath="shapes/dcat-ap-shacl.ttl",
        url=(
            f"https://raw.githubusercontent.com/SEMICeu/DCAT-AP/"
            f"{_DCAT_AP_COMMIT}/releases/3.0.0/shacl/dcat-ap-SHACL.ttl"
        ),
    ),
    # Vocabularies imported by the shapes ----------------------------------
    VendorEntry(relpath="vocab/dcat.ttl", url="https://www.w3.org/ns/dcat.ttl"),
    VendorEntry(relpath="vocab/dct.ttl", url="https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_terms.ttl"),
    VendorEntry(relpath="vocab/foaf.rdf", url="http://xmlns.com/foaf/spec/index.rdf"),
    VendorEntry(relpath="vocab/adms.ttl", url="https://www.w3.org/ns/adms.ttl"),
    # JSON-LD contexts -----------------------------------------------------
    VendorEntry(
        relpath="contexts/geodcat-ap.jsonld",
        url=(
            f"https://raw.githubusercontent.com/SEMICeu/GeoDCAT-AP/"
            f"{_SHAPES_COMMIT}/releases/3.0.0/context/geodcat-ap.jsonld"
        ),
    ),
    VendorEntry(relpath="contexts/dcat3.jsonld", url="https://www.w3.org/ns/dcat3.jsonld"),
    VendorEntry(relpath="contexts/dcat2.jsonld", url="https://www.w3.org/ns/dcat2.jsonld"),
]


# --- Fetch -----------------------------------------------------------------


def _client() -> httpx.Client:
    # rdflib serves Turtle when asked; raw.githubusercontent ignores Accept.
    return httpx.Client(
        timeout=HTTP_TIMEOUT_S,
        follow_redirects=True,
        headers={
            "User-Agent": USER_AGENT,
            # Asked for explicitly even when the server ignores it.
            "Accept": "text/turtle, application/rdf+xml, application/ld+json;q=0.9",
        },
    )


def _fetch(client: httpx.Client, entry: VendorEntry) -> tuple[bytes, str]:
    """Try `entry.url`, then each `alt_urls` in order. Returns the bytes
    and the URL that actually succeeded — useful for surfacing in
    PROVENANCE when a fallback wins.
    """
    urls = (entry.url, *entry.alt_urls)
    last_err: Exception | None = None
    for url in urls:
        try:
            resp = client.get(url)
            if resp.status_code == 404:
                last_err = httpx.HTTPStatusError(
                    f"404 at {url}", request=resp.request, response=resp,
                )
                if url != entry.url:
                    print(f"  alt {url} also 404", file=sys.stderr)
                continue
            resp.raise_for_status()
            if url != entry.url:
                print(f"  via alt URL {url}", file=sys.stderr)
            return resp.content, url
        except httpx.HTTPError as e:
            last_err = e
    raise last_err if last_err is not None else RuntimeError(f"no URL worked for {entry.relpath}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- PROVENANCE rewrite ----------------------------------------------------


_TABLE_HEADER_RE = re.compile(
    r"^\|\s*File\s*\|\s*Upstream URL\s*\|\s*SHA-256\s*\|\s*Retrieved \(UTC\)\s*\|\s*$",
    re.MULTILINE,
)


def _format_table(records: list[tuple[VendorEntry, str, str]]) -> str:
    lines = [
        "| File | Upstream URL | SHA-256 | Retrieved (UTC) |",
        "|---|---|---|---|",
    ]
    for entry, sha, retrieved in records:
        # Wrap the URL in backticks so markdown doesn't try to mangle the |.
        lines.append(f"| {entry.relpath} | `{entry.url}` | `{sha}` | {retrieved} |")
    return "\n".join(lines)


def _rewrite_provenance(records: list[tuple[VendorEntry, str, str]]) -> None:
    if not PROVENANCE.exists():
        raise FileNotFoundError(
            f"{PROVENANCE} is missing — the skeleton must be in tree before vendoring"
        )
    text = PROVENANCE.read_text(encoding="utf-8")
    header_match = _TABLE_HEADER_RE.search(text)
    if not header_match:
        raise RuntimeError(
            "could not find pinned-snapshot table header in PROVENANCE.md "
            "(expected '| File | Upstream URL | SHA-256 | Retrieved (UTC) |')"
        )
    # Snip from the header to the next blank line (the table) and replace
    # with the freshly-formatted one. Anything after the table is preserved.
    start = header_match.start()
    after = text[header_match.end():]
    blank_match = re.search(r"\n\s*\n", after)
    end_of_table = header_match.end() + (blank_match.start() if blank_match else len(after))
    new_table = _format_table(records)
    PROVENANCE.write_text(text[:start] + new_table + text[end_of_table:], encoding="utf-8")


# --- Main ------------------------------------------------------------------


def vendor() -> int:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    records: list[tuple[VendorEntry, str, str]] = []
    with _client() as client:
        for entry in _SOURCES:
            print(f"fetching {entry.relpath} ← {entry.url}")
            try:
                data, actual_url = _fetch(client, entry)
            except httpx.HTTPError as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                continue
            dest = VENDOR_DIR / entry.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            sha = _sha256(data)
            retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # Record the URL that actually returned the bytes — if a
            # fallback won, that's the one PROVENANCE should reflect.
            recorded = entry if actual_url == entry.url else VendorEntry(
                relpath=entry.relpath, url=actual_url,
            )
            records.append((recorded, sha, retrieved))
            print(f"  {len(data)} bytes, sha256={sha[:12]}…")
    _rewrite_provenance(records)
    return len(records)


def main() -> None:
    n = vendor()
    print(f"vendored {n}/{len(_SOURCES)} files under {VENDOR_DIR}")
    if n != len(_SOURCES):
        print("WARNING: some fetches failed; PROVENANCE.md only lists the successes",
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
