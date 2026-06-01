"""SHACL validation harness for the sidecar's eval suite.

Loads the vendored GeoDCAT-AP 3.0 shapes (plus DCAT-AP shapes and imported
vocabularies) once per pytest session and exposes a `validate(graph)`
function that returns a structured `ValidationResult`.

Design notes:

* `pyshacl` and `rdflib` live in `requirements-dev.txt` only — this module
  must never be imported from `src/`.
* The shapes graph is large (megabytes of Turtle once vocab imports load);
  parsing it on every test would make E1 useless. The
  `geodcat_ap_3_0_shapes` fixture below is session-scoped.
* `pyshacl` is invoked with `advanced=True` so SHACL-SPARQL constraints
  (GeoDCAT-AP uses some) actually run.
* Only `sh:Violation` counts as a failure; `sh:Warning` and `sh:Info` are
  collected separately so the conformance gate stays signal, and the
  paper's "warning taxonomy" footnote (Table 1) has the raw shape IDs.
* A `pyld` document loader override resolves the upstream URLs of the
  vendored JSON-LD contexts back to the local files, so client inputs that
  reference the official context URLs parse offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

VENDOR_ROOT = Path(__file__).resolve().parent / "fixtures" / "shacl" / "geodcat-ap-3.0"
SHAPES_DIR = VENDOR_ROOT / "shapes"
VOCAB_DIR = VENDOR_ROOT / "vocab"
CONTEXTS_DIR = VENDOR_ROOT / "contexts"


# Map upstream URLs ↔ local files so JSON-LD inputs validate offline.
# Keep this list in sync with eval/vendor_shapes.py:_SOURCES (contexts only).
_CONTEXT_URL_TO_FILE: dict[str, str] = {
    "https://www.w3.org/ns/dcat3.jsonld": "dcat3.jsonld",
    "https://www.w3.org/ns/dcat2.jsonld": "dcat2.jsonld",
    # The GeoDCAT-AP context — pyld will substitute the published location.
    "https://semiceu.github.io/GeoDCAT-AP/releases/3.0.0/context/geodcat-ap.jsonld": "geodcat-ap.jsonld",
    "https://raw.githubusercontent.com/SEMICeu/GeoDCAT-AP/master/releases/3.0.0/context/geodcat-ap.jsonld": "geodcat-ap.jsonld",
}


# --- pyld offline document loader ------------------------------------------


def install_offline_context_loader() -> None:
    """Make `pyld.jsonld` resolve the vendored contexts from disk.

    pyld is the JSON-LD parser rdflib uses. By default it issues HTTP GETs
    when a graph's `@context` is a URL. After this call, any URL in
    `_CONTEXT_URL_TO_FILE` is served from the local file instead.

    Idempotent: calling twice is harmless (we re-set the loader).
    """
    from pyld import jsonld as pyld  # type: ignore[import-not-found]

    def _loader(url: str, options=None):  # noqa: ANN001 — pyld's signature
        filename = _CONTEXT_URL_TO_FILE.get(url)
        if filename:
            path = CONTEXTS_DIR / filename
            return {
                "contextUrl": None,
                "documentUrl": url,
                "document": json.loads(path.read_text(encoding="utf-8")),
            }
        # Unknown URL — let the default loader handle it so failures are loud.
        return pyld.requests_document_loader()(url, options)

    pyld.set_document_loader(_loader)


# --- Shapes loader ---------------------------------------------------------


def _import_rdflib():
    try:
        import rdflib  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "rdflib is not installed; run `pip install -r requirements-dev.txt`"
        ) from e
    return rdflib


def _import_pyshacl():
    try:
        from pyshacl import validate as pyshacl_validate  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "pyshacl is not installed; run `pip install -r requirements-dev.txt`"
        ) from e
    return pyshacl_validate


@lru_cache(maxsize=1)
def load_shapes_graph():
    """Load the GeoDCAT-AP 3.0 SHACL shapes + DCAT-AP shapes into one graph.

    Cached for the process lifetime — call sites should not re-parse.
    Returns an `rdflib.Graph`.
    """
    rdflib = _import_rdflib()
    g = rdflib.Graph()
    shape_files = sorted(SHAPES_DIR.glob("*.ttl"))
    if not shape_files:
        raise RuntimeError(
            f"no SHACL shape files under {SHAPES_DIR}. Run "
            f"`python -m eval.vendor_shapes` first."
        )
    for f in shape_files:
        g.parse(str(f), format="turtle")
    return g


@lru_cache(maxsize=1)
def load_vocab_graph():
    """Load the imported vocabularies (DCAT, DCT, FOAF, ADMS), filtered
    to schema-only triples for use as pyshacl's `ont_graph`.

    pyshacl merges `ont_graph` into `data_graph` for inference and then
    validates the merged graph. The FOAF vocab in particular ships its
    own example bnodes (e.g. a `foaf:Person` with `org:memberOf` pointing
    at another bnode), and those examples get treated as our data —
    every record then fails the GeoDCAT-AP `org:memberOf` shape against
    those bnodes, not against ours.

    Keep only the schema-level edges the shapes actually need for
    inference: subClassOf / subPropertyOf chains, domain/range, and
    declarations of Classes/Properties. Drop everything else (instance
    data, examples, labels).

    Vendored files may not match their extension — xmlns.com serves
    FOAF as JSON-LD even when we ask for RDF/XML — so sniff first.
    """
    rdflib = _import_rdflib()
    raw = rdflib.Graph()
    for f in sorted(VOCAB_DIR.glob("*")):
        if f.name.startswith("."):
            continue
        fmt = _sniff_rdf_format(f)
        try:
            raw.parse(str(f), format=fmt)
        except Exception as e:  # noqa: BLE001 — vocab parse problems are loud
            raise RuntimeError(f"failed to parse vocab {f} as {fmt}: {e}") from e
    return _schema_only(raw, rdflib)


def _schema_only(g, rdflib_mod):
    """Return a copy of `g` with only schema-relevant triples.

    Schema-relevant means: edges that drive rdfs:subClass / subProperty
    inference, declarations of Classes and Properties, and
    domain/range. Anything else (vocab example bnodes, labels, comments,
    seeAlso links) gets dropped so pyshacl doesn't validate it.
    """
    URIRef = rdflib_mod.URIRef
    RDF = rdflib_mod.RDF
    RDFS = rdflib_mod.RDFS
    OWL = rdflib_mod.Namespace("http://www.w3.org/2002/07/owl#")
    schema_preds = {
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        RDFS.domain,
        RDFS.range,
    }
    schema_types = {
        RDFS.Class,
        RDF.Property,
        URIRef(str(OWL) + "Class"),
        URIRef(str(OWL) + "ObjectProperty"),
        URIRef(str(OWL) + "DatatypeProperty"),
        URIRef(str(OWL) + "AnnotationProperty"),
    }
    out = rdflib_mod.Graph()
    for s, p, o in g:
        if p in schema_preds:
            out.add((s, p, o))
        elif p == RDF.type and o in schema_types:
            out.add((s, p, o))
    return out


def _sniff_rdf_format(path: Path) -> str:
    """Return an rdflib parser name for `path` based on content, not
    extension. Falls back to the extension when the head is ambiguous.
    """
    head = path.read_bytes()[:512].lstrip()
    if head.startswith(b"{") or head.startswith(b"["):
        return "json-ld"
    if head.startswith(b"<?xml") or head.startswith(b"<rdf:") or head.startswith(b"<RDF") or head.startswith(b"<!DOCTYPE"):
        return "xml"
    if head.startswith(b"@prefix") or head.startswith(b"@base") or head.startswith(b"PREFIX"):
        return "turtle"
    return {
        ".jsonld": "json-ld",
        ".json": "json-ld",
        ".rdf": "xml",
        ".xml": "xml",
        ".ttl": "turtle",
    }.get(path.suffix.lower(), "turtle")


# --- Validation surface ----------------------------------------------------


SH = "http://www.w3.org/ns/shacl#"
_SEVERITY_VIOLATION = SH + "Violation"
_SEVERITY_WARNING = SH + "Warning"
_SEVERITY_INFO = SH + "Info"


@dataclass
class ValidationFinding:
    severity: str         # short tag: "Violation" | "Warning" | "Info" | "Unknown"
    source_shape: str     # IRI of the sh:NodeShape / sh:PropertyShape, or ""
    focus_node: str       # IRI of the offending node, or ""
    path: str             # IRI of the property path, or ""
    message: str          # rendered sh:resultMessage, or ""


@dataclass
class ValidationResult:
    conforms: bool        # True iff no Violations (Warnings ignored, per design)
    violations: list[ValidationFinding] = field(default_factory=list)
    warnings: list[ValidationFinding] = field(default_factory=list)
    infos: list[ValidationFinding] = field(default_factory=list)
    text: str = ""        # pyshacl's human-readable report (debugging only)


def _severity_tag(iri: str) -> str:
    return {
        _SEVERITY_VIOLATION: "Violation",
        _SEVERITY_WARNING: "Warning",
        _SEVERITY_INFO: "Info",
    }.get(iri, "Unknown")


def _findings_from_results(results_graph) -> list[ValidationFinding]:
    rdflib = _import_rdflib()
    ns = rdflib.Namespace(SH)
    findings: list[ValidationFinding] = []
    for vr in results_graph.subjects(rdflib.RDF.type, ns.ValidationResult):
        severity = next(results_graph.objects(vr, ns.resultSeverity), None)
        source = next(results_graph.objects(vr, ns.sourceShape), None)
        focus = next(results_graph.objects(vr, ns.focusNode), None)
        path = next(results_graph.objects(vr, ns.resultPath), None)
        msg = next(results_graph.objects(vr, ns.resultMessage), None)
        findings.append(
            ValidationFinding(
                severity=_severity_tag(str(severity) if severity else ""),
                source_shape=str(source) if source else "",
                focus_node=str(focus) if focus else "",
                path=str(path) if path else "",
                message=str(msg) if msg else "",
            )
        )
    return findings


def validate(graph: Union["object", dict, str]) -> ValidationResult:
    """Validate `graph` against the vendored GeoDCAT-AP 3.0 shapes.

    Accepts:
      * an `rdflib.Graph` (used as-is);
      * a `dict` (treated as JSON-LD; parsed with the offline loader);
      * a `str` path to a `.jsonld` / `.ttl` / `.rdf` file.

    Returns a `ValidationResult` separating Violations from Warnings/Infos.
    `conforms` is True iff `len(violations) == 0` — Warnings do not fail.
    """
    rdflib = _import_rdflib()
    pyshacl_validate = _import_pyshacl()

    data_graph = _coerce_to_graph(graph, rdflib)
    conforms, results_graph, text = pyshacl_validate(
        data_graph=data_graph,
        shacl_graph=load_shapes_graph(),
        ont_graph=load_vocab_graph(),
        advanced=True,                # enable SHACL-SPARQL constraints
        inference="rdfs",             # subclassing imports activate properly
        meta_shacl=False,
        debug=False,
    )
    findings = _findings_from_results(results_graph)
    violations = [f for f in findings if f.severity == "Violation"]
    warnings = [f for f in findings if f.severity == "Warning"]
    infos = [f for f in findings if f.severity == "Info"]
    return ValidationResult(
        conforms=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        infos=infos,
        text=text,
    )


def _coerce_to_graph(value, rdflib_mod):  # noqa: ANN001 — rdflib type at runtime
    Graph = rdflib_mod.Graph
    if isinstance(value, Graph):
        return value
    if isinstance(value, dict):
        install_offline_context_loader()
        g = Graph()
        g.parse(data=json.dumps(value), format="json-ld")
        return g
    if isinstance(value, (str, Path)):
        path = Path(value)
        fmt = {
            ".jsonld": "json-ld",
            ".json": "json-ld",
            ".ttl": "turtle",
            ".rdf": "xml",
            ".xml": "xml",
        }.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"unknown RDF extension for {path}")
        if fmt == "json-ld":
            install_offline_context_loader()
        g = Graph()
        g.parse(str(path), format=fmt)
        return g
    raise TypeError(
        f"validate() accepts rdflib.Graph, dict (JSON-LD), or path; got {type(value).__name__}"
    )


# --- pytest fixture --------------------------------------------------------

try:
    import pytest  # type: ignore[import-not-found]
except ImportError:  # the module is also usable from `eval/run_e1_conformance.py`
    pytest = None  # type: ignore[assignment]


if pytest is not None:

    @pytest.fixture(scope="session")
    def geodcat_ap_3_0_shapes():
        """Session-scoped: load the shapes graph once.

        The fixture exposes a small dict so individual tests can choose
        whether to call `validate(...)` directly or inspect the underlying
        rdflib graphs (for debugging shape failures).
        """
        return {
            "shapes": load_shapes_graph(),
            "vocab": load_vocab_graph(),
            "validate": validate,
        }
