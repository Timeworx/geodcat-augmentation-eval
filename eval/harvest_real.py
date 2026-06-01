"""Harvest 40 real GeoDCAT-AP records from data.europa.eu (Task 1.2).

Network note: data.europa.eu is not on the CI/sandbox allow-list. Run this
script once from an internet-connected workstation, commit the contents of
`eval/corpus/real/` and the appended manifest rows, and treat the harvest as
a versioned snapshot so all downstream experiments are reproducible offline.

Output layout (per record):
    eval/corpus/real/<slug>.raw.<ext>     raw fetched payload (audit trail)
    eval/corpus/real/<slug>.jsonld        normalised JSON-LD (E1/E2 input)

Manifest: appends one row per harvested record to
`eval/corpus/manifest.csv`. The dedupe key is `source_url`, so re-runs do
not duplicate rows.

Strategy:
    * Use the CKAN-style search API at
      `https://data.europa.eu/api/hub/search/datasets` with a geospatial
      filter. (The SPARQL endpoint at `data.europa.eu/sparql` is the
      alternative; the CKAN one is simpler and adequate for this sample.)
    * Filter for datasets carrying a geospatial signal — INSPIRE category
      / `spatial` field present / themed under environment or transport.
    * For each candidate, content-negotiate `application/ld+json` from the
      record's RDF endpoint. Fall back to `text/turtle` + rdflib conversion.
    * Skip records that don't parse to a `dcat:Dataset` node; the goal is
      ecological validity, not 40 records at any cost. Stop at 40 OK.

Run:
    python -m eval.harvest_real          # default: 40 records
    python -m eval.harvest_real --n 5    # smaller smoke run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = REPO_ROOT / "eval" / "corpus" / "real"
MANIFEST = REPO_ROOT / "eval" / "corpus" / "manifest.csv"

# data.europa.eu's hub search has migrated from a GET-with-facets surface
# to a POST + JSON-body shape. The GET variant returns 400 for any
# non-empty `facets` param now. We POST against /search/search; the
# response is the same `{result: {results: [...]}}` shape.
SEARCH_URL = "https://data.europa.eu/api/hub/search/search"
# Per-dataset RDF fetch: the canonical hub URL serves content-negotiated
# JSON-LD when given Accept: application/ld+json.
REPO_BASE_URL = "https://data.europa.eu/data/datasets"

# data.europa.eu's hub no longer exposes a `categories` facet (the
# available facets are dataScope, is_hvd, source_type, hvdCategory,
# country, catalog, superCatalog, …). Default to no facet — the
# eval corpus benefits from ecological diversity more than from a
# geospatial pre-filter that may be stale. Pass `--categories=...`
# explicitly to opt back in once the right facet ID is known.
_DEFAULT_CATEGORY_FACETS: list[str] = []

USER_AGENT = "timeworx-sidecar-eval/0.1 (+https://timeworx.io)"
HTTP_TIMEOUT_S = 30.0
PAGE_PAUSE_S = 0.25  # gentle to the search backend

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


# --- HTTP plumbing ---------------------------------------------------------


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=HTTP_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
    )


def _search_body(
    *,
    page: int,
    limit: int,
    categories: list[str] | None,
    sort: list[str],
) -> dict:
    """Body shape for POST /api/hub/search/search.

    `page` is 0-indexed. `filters: ["dataset"]` scopes the search to
    datasets (the same endpoint can return publishers, distributions).
    `sort` varies between sweeps so we can collect more than one
    "first 10" worth of records — the hub caps `limit` server-side at
    10 and ignores `page` in our calls, so cycling through sort orders
    is the most reliable way to broaden coverage.
    """
    body: dict = {
        "limit": limit,
        "page": page,
        "filters": ["dataset"],
        "facetOperator": "AND",
        "facets": {},
        "facetFilters": [],
        "showScore": False,
        "sort": list(sort),
    }
    if categories:
        body["facets"]["categories"] = list(categories)
    return body


# Each sweep is one POST. The hub returns ~10 results regardless of
# `limit`, so to harvest the brief's 30–50 target we cycle through
# distinct sort orders — each one's "first 10" intersects only
# loosely with the others. Order matters: most informative orders
# first so a short run still gets useful diversity.
_SORT_SWEEPS: list[list[str]] = [
    ["relevance+desc", "modified+desc"],
    ["modified+desc"],
    ["modified+asc"],
    ["issued+desc"],
    ["issued+asc"],
    ["title+asc"],
    ["title+desc"],
    ["relevance+desc"],
]


def _search(
    client: httpx.Client,
    *,
    page: int,
    limit: int,
    categories: list[str] | None,
    sort: list[str],
) -> dict:
    body = _search_body(page=page, limit=limit, categories=categories, sort=sort)
    resp = client.post(
        SEARCH_URL,
        json=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    if resp.status_code >= 400:
        # Surface enough context to recover. The hub's error bodies usually
        # name the offending field directly.
        raise RuntimeError(
            f"search POST {SEARCH_URL} failed: HTTP {resp.status_code} "
            f"body={resp.text[:300]!r} request_body={json.dumps(body)[:200]}"
        )
    return resp.json()


def _extract_results(body: dict) -> list[dict]:
    """Pull the dataset list out of the search response, tolerating a few
    plausible response shapes — the hub has rotated keys before and a
    silent shape change is the most likely explanation for "0 harvested
    with no errors".
    """
    if not isinstance(body, dict):
        return []
    for path in (
        ("result", "results"),
        ("result", "datasets"),
        ("result", "items"),
        ("data", "results"),
        ("data", "datasets"),
        ("results",),
        ("datasets",),
    ):
        cursor: object = body
        ok = True
        for key in path:
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                ok = False
                break
        if ok and isinstance(cursor, list):
            return cursor
    return []


def _iter_candidates(
    client: httpx.Client,
    *,
    categories: list[str] | None,
    debug: bool = False,
    sweeps: list[list[str]] | None = None,
) -> Iterator[dict]:
    """Yield dataset summaries from the hub search across multiple sort
    sweeps. Falls back to an unfiltered (no-category) search if the
    faceted request returns 400 — the hub occasionally renames category
    codes, so a graceful degrade is better than a hard fail mid-harvest.
    """
    cats: list[str] | None = list(categories) if categories else None
    sort_orders = sweeps or _SORT_SWEEPS
    for attempt in (cats, None) if cats else (None,):
        try:
            seen_any = False
            for sweep_idx, sort_order in enumerate(sort_orders):
                body = _search(
                    client, page=0, limit=50, categories=attempt, sort=sort_order,
                )
                results = _extract_results(body)
                count_total = _extract_count(body)
                if debug:
                    top_keys = list(body.keys()) if isinstance(body, dict) else type(body).__name__
                    print(
                        f"  sweep {sweep_idx} sort={sort_order} categories={attempt} -> "
                        f"{len(results)} results (count={count_total}, "
                        f"top-level keys: {top_keys})",
                        file=sys.stderr,
                    )
                    if sweep_idx == 0 and not results:
                        snippet = json.dumps(body)[:600] if isinstance(body, dict) else repr(body)[:600]
                        print(f"  response snippet: {snippet}", file=sys.stderr)
                if not results:
                    continue
                seen_any = True
                for r in results:
                    yield r
                time.sleep(PAGE_PAUSE_S)
            if seen_any:
                return
            if attempt is None:
                return
        except RuntimeError as e:
            if attempt is None:
                raise
            print(
                f"categories filter {attempt} rejected: {e}; "
                f"retrying without facets",
                file=sys.stderr,
            )


def _extract_count(body: dict) -> int | None:
    """Pull the total-result count from the search response so debug
    output can compare available-vs-fetched."""
    if not isinstance(body, dict):
        return None
    for path in (("result", "count"), ("count",), ("total",), ("result", "total")):
        cursor: object = body
        for key in path:
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                cursor = None
                break
        if isinstance(cursor, int):
            return cursor
    return None


# --- Record extraction -----------------------------------------------------


# --- Search-record → GeoDCAT-AP JSON-LD converter -------------------------

# The hub's `/data/datasets/{id}` URL serves only HTML now (it ignores
# Accept: application/ld+json), so the canonical RDF the old code tried
# to content-negotiate is unreachable. Fortunately, the *search response*
# records already carry every field GeoDCAT-AP would: title, description,
# distributions, publisher, keywords, country, identifier, themes,
# contact_point, issued/modified dates. We treat the search response as
# the source of truth and wrap it in a JSON-LD envelope here.


def _coerce_lang_map(value):
    """data.europa.eu surfaces title/description as a per-language map
    `{"en": "...", "fr": "..."}`. Pick English first, then any present
    language. Returns a language-tagged JSON-LD literal, or None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return {"@language": "en", "@value": value}
    if isinstance(value, dict):
        if "en" in value and value["en"]:
            return {"@language": "en", "@value": value["en"]}
        for lang, v in value.items():
            if v:
                return {"@language": lang, "@value": v}
    return None


def _resource_url(node: dict) -> Optional[str]:
    """A distribution-shaped dict can carry the download URL under any
    of access_url / download_url / url. Pick the first present."""
    for key in ("access_url", "download_url", "url"):
        v = node.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            inner = v.get("resource") or v.get("@id") or v.get("uri")
            if isinstance(inner, str) and inner:
                return inner
    return None


def _to_distribution(dist: dict) -> Optional[dict]:
    if not isinstance(dist, dict):
        return None
    access = _resource_url(dist)
    if not access:
        return None
    node = {
        "@type": "dcat:Distribution",
        "dcat:accessURL": {"@id": access, "@type": "rdfs:Resource"},
    }
    title = _coerce_lang_map(dist.get("title"))
    if title:
        node["dct:title"] = title
    fmt = dist.get("format")
    if isinstance(fmt, dict):
        fmt_id = fmt.get("resource") or fmt.get("@id")
        if fmt_id:
            node["dct:format"] = {"@id": fmt_id, "@type": "dct:MediaTypeOrExtent"}
    elif isinstance(fmt, str) and fmt:
        node["dct:format"] = {"@value": fmt}
    return node


def _to_theme(theme) -> Optional[dict]:
    """Themes come in two shapes: a string URI, or
    {"resource": "<uri>", "title": "...", ...}.
    """
    if isinstance(theme, str):
        return {"@id": theme, "@type": "skos:Concept"}
    if isinstance(theme, dict):
        uri = theme.get("resource") or theme.get("uri") or theme.get("id")
        label = _coerce_lang_map(theme.get("title") or theme.get("label"))
        node: dict = {"@type": "skos:Concept"}
        if isinstance(uri, str):
            node["@id"] = uri
        if label:
            node["skos:prefLabel"] = label
        if node.get("@id") or node.get("skos:prefLabel"):
            return node
    return None


def _to_publisher(publisher) -> Optional[dict]:
    if isinstance(publisher, str):
        return {"@id": publisher, "@type": "foaf:Agent"}
    if isinstance(publisher, dict):
        node: dict = {"@type": ["foaf:Organization", "foaf:Agent"]}
        uri = publisher.get("resource") or publisher.get("@id") or publisher.get("uri")
        if isinstance(uri, str):
            node["@id"] = uri
        name = _coerce_lang_map(publisher.get("name") or publisher.get("title"))
        if name:
            node["foaf:name"] = name
        return node
    return None


def _to_keywords(keywords) -> list:
    flat: list = []
    if isinstance(keywords, dict):
        # Per-language buckets: {"en": ["a", "b"], "de": [...]}
        for lang, items in keywords.items():
            if isinstance(items, list):
                for kw in items:
                    if isinstance(kw, str) and kw:
                        flat.append({"@language": lang, "@value": kw})
    elif isinstance(keywords, list):
        for kw in keywords:
            if isinstance(kw, str) and kw:
                flat.append({"@language": "en", "@value": kw})
            elif isinstance(kw, dict):
                node = _coerce_lang_map(kw.get("title") or kw.get("name") or kw)
                if node:
                    flat.append(node)
    return flat


_STD_KEYS = {
    "id", "name", "title", "description", "keywords", "publisher",
    "distributions", "resource", "country", "countries", "theme",
    "themes", "identifier", "index", "spatial", "contact_point",
    "issued", "modified", "translation_meta",
}


def _record_to_jsonld(record: dict, *, slug: str) -> dict:
    """Wrap a data.europa.eu search-response record in a GeoDCAT-AP
    JSON-LD envelope. Known fields map to standard predicates; unknown
    fields pass through under a non-standard `deuropa:` namespace so
    E2's zero-loss check can verify they survive the augmenter.
    """
    record_id = record.get("id") or record.get("name") or slug
    dataset_node: dict = {
        "@id": f"https://data.europa.eu/data/datasets/{record_id}",
        "@type": "dcat:Dataset",
        "dct:identifier": record.get("identifier") or record_id,
    }
    title = _coerce_lang_map(record.get("title"))
    if title:
        dataset_node["dct:title"] = title
    description = _coerce_lang_map(record.get("description"))
    if description:
        dataset_node["dct:description"] = description

    publisher = _to_publisher(record.get("publisher"))
    if publisher:
        dataset_node["dct:publisher"] = publisher

    keywords = _to_keywords(record.get("keywords"))
    if keywords:
        dataset_node["dcat:keyword"] = keywords

    themes_raw = record.get("theme") or record.get("themes") or []
    if isinstance(themes_raw, dict):
        themes_raw = [themes_raw]
    themes: list = []
    if isinstance(themes_raw, list):
        for t in themes_raw:
            n = _to_theme(t)
            if n:
                themes.append(n)
    if themes:
        dataset_node["dcat:theme"] = themes

    spatial_label = record.get("country") or record.get("countries")
    if isinstance(spatial_label, list) and spatial_label:
        spatial_label = spatial_label[0]
    if isinstance(spatial_label, dict):
        spatial_label = spatial_label.get("name") or spatial_label.get("title") or spatial_label.get("id")
    if isinstance(spatial_label, str) and spatial_label:
        dataset_node["dct:spatial"] = {
            "@type": ["dct:Location", "skos:Concept"],
            "skos:prefLabel": {"@language": "en", "@value": spatial_label},
        }

    dists_raw = record.get("distributions") or record.get("resource") or []
    if isinstance(dists_raw, dict):
        dists_raw = [dists_raw]
    dists: list = []
    if isinstance(dists_raw, list):
        for d in dists_raw:
            n = _to_distribution(d)
            if n:
                dists.append(n)
    if dists:
        dataset_node["dcat:distribution"] = dists

    # Anything else rides along under a non-standard namespace. E2's
    # zero-loss check needs unknown predicates on the input so it can
    # verify they survive augmentation.
    for key, value in record.items():
        if key in _STD_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        dataset_node[f"deuropa:{key}"] = value

    return {
        "@context": {
            "dcat": "http://www.w3.org/ns/dcat#",
            "dct": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "geo": "http://www.opengis.net/ont/geosparql#",
            "deuropa": "https://data.europa.eu/api/ext#",
        },
        "@graph": [dataset_node],
    }


_GEO_HINTS = ("spatial", "geographicBoundingBox", "place", "bbox")


def _looks_geospatial(record: dict) -> bool:
    """Heuristic: the record advertises something geospatial. data.europa.eu
    surfaces this in multiple shapes — themes, spatial fields, or INSPIRE
    categorisation."""
    if any(record.get(k) for k in _GEO_HINTS):
        return True
    themes = record.get("themes") or record.get("theme") or []
    if isinstance(themes, list) and any(
        isinstance(t, str) and ("envi" in t.lower() or "agri" in t.lower())
        for t in themes
    ):
        return True
    return False


_LD_JSON_MEDIATYPES = ("application/ld+json", "application/json+ld")
_TURTLE_MEDIATYPES = ("text/turtle", "application/x-turtle")
_RDF_XML_MEDIATYPES = ("application/rdf+xml",)


def _rdf_endpoint(record: dict) -> Optional[str]:
    """data.europa.eu mints a canonical RDF URL per dataset under
    `/data/datasets/<id>`. Some records also carry resources that point at
    the source publisher's own RDF — prefer the canonical one."""
    cid = record.get("id") or record.get("name")
    if not cid:
        return None
    return f"{REPO_BASE_URL}/{cid}"


# Single Accept header listing every shape we can ingest, in priority
# order. Letting the server pick a format is more reliable than running
# two sequential GETs — some hubs ignore the second Accept entirely
# after the first fetch loads a session.
_ACCEPT_RDF = ", ".join([
    "application/ld+json",
    "application/json+ld",
    "application/json;q=0.9",  # some hubs label JSON-LD as plain JSON
    "text/turtle;q=0.8",
    "application/rdf+xml;q=0.7",
])

_fetch_debug_done = False  # one-time payload dump under --debug


def _fetch_rdf(
    client: httpx.Client, url: str, *, debug: bool = False
) -> tuple[bytes, str]:
    """Single content-negotiated GET. Returns (bytes, media_type).

    Logs the first response's status + content-type + payload prefix
    under `debug` so a recurring parse failure can be diagnosed without
    instrumenting the client.
    """
    global _fetch_debug_done
    resp = client.get(url, headers={"Accept": _ACCEPT_RDF})
    ct = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if debug and not _fetch_debug_done:
        print(
            f"  fetch sample {url}\n"
            f"    HTTP {resp.status_code} content-type={ct!r} "
            f"length={len(resp.content)}",
            file=sys.stderr,
        )
        print(f"    payload[:300]={resp.content[:300]!r}", file=sys.stderr)
        _fetch_debug_done = True
    resp.raise_for_status()
    if not ct:
        # Some hubs respond 200 with no Content-Type. Best-effort sniff
        # so we don't drop the record on a missing header alone.
        text = resp.content[:64].lstrip()
        if text.startswith(b"{") or text.startswith(b"["):
            ct = "application/json"
        elif text.startswith(b"<"):
            ct = "application/rdf+xml"
        else:
            ct = "text/turtle"
    return resp.content, ct


def _to_jsonld(payload: bytes, media_type: str) -> dict:
    """Normalise the fetched payload to a JSON-LD-compatible dict.

    Strategy:
      1. JSON / JSON-LD → `json.loads`. Many endpoints serve JSON-LD
         labelled as plain `application/json`; we still try `json.loads`
         when the byte prefix looks JSON-shaped.
      2. Turtle / RDF-XML → rdflib parse, then convert to JSON-LD via
         N-Quads + pyld. This sidesteps rdflib 7.x's built-in JSON-LD
         serializer, which raises `expected str instance, NoneType
         found` when a literal carries an unset datatype on certain
         graphs the hub emits.

    Returns a dict always — list-shaped JSON-LD is wrapped in `@graph`.
    """
    text_head = payload[:64].lstrip()
    looks_json = text_head.startswith(b"{") or text_head.startswith(b"[")
    if media_type in _LD_JSON_MEDIATYPES or media_type == "application/json" or looks_json:
        try:
            doc = json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"payload labelled {media_type!r} did not parse as JSON: {e}"
            ) from e
        if isinstance(doc, list):
            return {"@graph": doc}
        if isinstance(doc, dict):
            return doc
        raise RuntimeError(f"unexpected JSON-LD shape: {type(doc).__name__}")

    try:
        import rdflib  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "rdflib is required for non-JSON responses; install dev deps "
            "(`pip install -r requirements-dev.txt`)"
        ) from e
    g = rdflib.Graph()
    fmt = "turtle" if media_type in _TURTLE_MEDIATYPES else "xml"
    g.parse(data=payload, format=fmt)
    return _serialize_via_nquads_to_jsonld(g)


def _serialize_via_nquads_to_jsonld(graph) -> dict:
    """rdflib graph → N-Quads → pyld `from_rdf` → JSON-LD expanded form
    wrapped in `@graph`. pyld is already a transitive dep (via pyshacl).

    Avoids the rdflib 7.x bug where serialise(format="json-ld") crashes
    with `expected str instance, NoneType found` on literals with no
    datatype.
    """
    nq = graph.serialize(format="nquads")
    if isinstance(nq, bytes):
        nq = nq.decode("utf-8")
    try:
        from pyld import jsonld as pyld  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "pyld is required to convert RDF → JSON-LD reliably; install "
            "it via `pip install -r requirements-dev.txt` (pulled in "
            "transitively by pyshacl)"
        ) from e
    expanded = pyld.from_rdf(nq, {"format": "application/n-quads"})
    return {"@graph": expanded}


_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _slug_for(record: dict, source_url: str) -> str:
    base = record.get("id") or record.get("name") or hashlib.sha1(source_url.encode()).hexdigest()[:12]
    slug = _SLUG_RE.sub("-", str(base).lower()).strip("-")
    return slug[:80] or hashlib.sha1(source_url.encode()).hexdigest()[:12]


def _ext_for(media_type: str) -> str:
    return {
        "application/ld+json": "jsonld",
        "application/json+ld": "jsonld",
        "text/turtle": "ttl",
        "application/x-turtle": "ttl",
        "application/rdf+xml": "rdf",
    }.get(media_type, "bin")


def _richness_tier(graph: dict) -> str:
    """Cheap tier heuristic so the manifest carries usable buckets.
    Rich = dct:title + dct:publisher + dct:spatial all present on some node.
    Partial = at least dct:title. Otherwise minimal."""
    nodes = []
    if isinstance(graph, dict):
        if "@graph" in graph and isinstance(graph["@graph"], list):
            nodes = [n for n in graph["@graph"] if isinstance(n, dict)]
        else:
            nodes = [graph]
    flat_keys: set[str] = set()
    for n in nodes:
        flat_keys.update(k.split(":")[-1] if ":" in k else k for k in n.keys())
    has_title = "title" in flat_keys
    has_publisher = "publisher" in flat_keys
    has_spatial = "spatial" in flat_keys
    if has_title and has_publisher and has_spatial:
        return "rich"
    if has_title:
        return "partial"
    return "minimal"


# --- Manifest dedupe -------------------------------------------------------


def _existing_source_urls() -> set[str]:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {row["source_url"] for row in csv.DictReader(f) if row.get("source_url")}


def _append_manifest(rows: list[dict]) -> None:
    new_file = not MANIFEST.exists()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# --- Main loop -------------------------------------------------------------


def harvest(
    target: int = 40,
    *,
    categories: list[str] | None = None,
    debug: bool = False,
    accept_any: bool = True,
) -> int:
    """Harvest `target` real records.

    `accept_any` defaults to True — the geospatial heuristic was tuned
    against an older API surface and rejects most current records. For
    the eval corpus's "ecological validity" goal, accepting every
    candidate is more honest than filtering on stale assumptions; E1
    and E2 will surface any non-geospatial outliers via their warning
    taxonomies.
    """
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    seen = _existing_source_urls()
    rows: list[dict] = []
    accepted = 0
    sample_keys_dumped = False
    # Diagnostic counters — surface where records get filtered out so a
    # zero-record run is not a silent dead end.
    n_candidates = 0
    n_not_geo = 0
    n_no_source = 0
    n_dupe = 0
    n_fetch_fail = 0
    n_parse_fail = 0
    with _client() as client:
        for record in _iter_candidates(client, categories=categories, debug=debug):
            n_candidates += 1
            if debug and not sample_keys_dumped and isinstance(record, dict):
                print(
                    f"  sample candidate keys: {sorted(record.keys())}",
                    file=sys.stderr,
                )
                sample_keys_dumped = True
            if accepted >= target:
                break
            if not accept_any and not _looks_geospatial(record):
                n_not_geo += 1
                continue
            source_url = _rdf_endpoint(record)
            if not source_url:
                n_no_source += 1
                continue
            if source_url in seen:
                n_dupe += 1
                continue
            slug = _slug_for(record, source_url)
            try:
                jsonld = _record_to_jsonld(record, slug=slug)
            except Exception as e:  # noqa: BLE001
                n_parse_fail += 1
                print(f"skip {source_url} (convert): {e}", file=sys.stderr)
                continue
            raw_path = REAL_DIR / f"{slug}.raw.json"
            norm_path = REAL_DIR / f"{slug}.jsonld"
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
                f.write("\n")
            with norm_path.open("w", encoding="utf-8") as f:
                json.dump(jsonld, f, indent=2, ensure_ascii=False)
                f.write("\n")
            rows.append(
                {
                    "id": f"real_{slug}",
                    "path": str(norm_path.relative_to(REPO_ROOT)),
                    "origin": "real",
                    "richness_tier": _richness_tier(jsonld),
                    "expected_path": "ai_ready",
                    "known_input_issues": "",  # E1 will populate via the warning taxonomy
                    "source_url": source_url,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            seen.add(source_url)
            accepted += 1
            time.sleep(PAGE_PAUSE_S)
    _append_manifest(rows)
    print(
        f"harvest funnel: candidates={n_candidates} "
        f"not_geospatial={n_not_geo} no_source_url={n_no_source} "
        f"already_seen={n_dupe} fetch_failed={n_fetch_fail} "
        f"parse_failed={n_parse_fail} accepted={accepted}",
        file=sys.stderr,
    )
    return accepted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=40, help="target record count (default 40)")
    p.add_argument(
        "--categories",
        default=",".join(_DEFAULT_CATEGORY_FACETS),
        help=(
            "comma-separated category facet values to filter the search on "
            "(default: empty — the hub no longer exposes the `categories` "
            "facet that the old harvester relied on, so unfiltered firehose "
            "is more reliable)."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="print search response shape + per-page result counts to stderr",
    )
    p.add_argument(
        "--geospatial-only",
        action="store_true",
        help=(
            "apply the geospatial heuristic — only accept candidates that "
            "advertise a spatial signal. Default is off (corpus diversity "
            "matters more than filter precision)."
        ),
    )
    args = p.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    n = harvest(
        target=args.n,
        categories=cats,
        debug=args.debug,
        accept_any=not args.geospatial_only,
    )
    print(f"harvested {n} real records -> {REAL_DIR}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
