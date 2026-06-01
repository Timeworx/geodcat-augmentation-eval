from .types import SegmentationType

TIMEWORX_DATASET_NS = "https://timeworx.io/datasets"
TIMEWORX_PUBLISHER_ID = "https://timeworx.io/organization"
TIMEWORX_PUBLISHER_NAME = "Timeworx.io"

COCO_FORMAT_IRI = "https://www.iana.org/assignments/media-types/application/json"
COCO_STANDARD_TITLE = "MS COCO Object Detection Format"

OUTPUT_CONTEXT_PREFIXES = {
    # Prefixes the augmenter emits unconditionally. Minimal client inputs
    # drop most of the standard GeoDCAT-AP context; we re-fill anything
    # we actually serialise so the output JSON-LD parses cleanly even
    # against a stripped-down input.
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov": "http://www.w3.org/ns/prov#",
    "spdx": "http://spdx.org/rdf/terms#",
    "locn": "http://www.w3.org/ns/locn#",
    "org": "http://www.w3.org/ns/org#",
}

PROCESSING_KEYWORDS: dict[SegmentationType, tuple[str, ...]] = {
    SegmentationType.INSTANCE: ("Instance Segmentation", "Plant Counting"),
    SegmentationType.SEMANTIC: ("Semantic Segmentation",),
    SegmentationType.PANOPTIC: ("Panoptic Segmentation", "Advanced Phenotyping"),
}


def timeworx_publisher() -> dict:
    # GeoDCAT-AP 3.0's foaf:AgentShape requires `org:Organization` on
    # any dct:publisher. Leaving `foaf:Organization` + `foaf:Agent` in
    # too — they're harmless additions and keep older DCAT-AP consumers
    # happy.
    return {
        "@id": TIMEWORX_PUBLISHER_ID,
        "@type": ["org:Organization", "foaf:Organization", "foaf:Agent"],
        "foaf:name": TIMEWORX_PUBLISHER_NAME,
    }


def coco_distribution(dataset_iri: str, access_url: str) -> dict:
    seg_type_label = "Instance Annotations (COCO Format)"
    return {
        "@id": f"{dataset_iri}#dist-coco",
        "@type": "dcat:Distribution",
        "dct:title": seg_type_label,
        "dcat:accessURL": {"@id": access_url, "@type": "rdfs:Resource"},
        "dct:format": {"@id": COCO_FORMAT_IRI, "@type": "dct:MediaTypeOrExtent"},
        "dct:conformsTo": {"@type": "dct:Standard", "dct:title": COCO_STANDARD_TITLE},
    }
