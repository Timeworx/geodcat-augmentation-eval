from . import jsonld, vocab
from .types import SegmentationType


def augment_ai_ready(
    base_graph: dict,
    *,
    job_id: str,
    segmentation_type: SegmentationType,
    coco_access_url: str,
) -> dict:
    """Augment a client GeoDCAT-AP 3.0 graph (AI-Ready path).

    Zero-loss invariant: every predicate on the input dataset node survives
    on the output, except those explicitly replaced (@id, dct:identifier,
    dct:title, dct:publisher, dcat:distribution). prov:wasDerivedFrom is
    added; processing keywords are appended to dcat:keyword (deduped).
    """
    cloned = jsonld.clone(base_graph)
    dataset = jsonld.find_dataset(cloned)

    original_id = dataset.get("@id")
    original_title = dataset.get("dct:title")
    original_description = dataset.get("dct:description")

    dataset_iri = f"{vocab.TIMEWORX_DATASET_NS}/{job_id}"
    dataset["@id"] = dataset_iri
    dataset["dct:identifier"] = job_id
    dataset["dct:title"] = _augmented_title(original_title, segmentation_type)
    # GeoDCAT-AP 3.0 requires dct:description minCount=1. Carry the
    # original through when present, fall back to a derived description
    # when the input had none — keeps the augmenter conformant on
    # minimal-context inputs without inventing client metadata.
    dataset["dct:description"] = _augmented_description(
        original_description, segmentation_type,
    )
    dataset["dct:publisher"] = vocab.timeworx_publisher()

    for kw in vocab.PROCESSING_KEYWORDS[segmentation_type]:
        jsonld.append_keyword(dataset, kw)

    derived_from: dict = {"@id": original_id}
    if original_title is not None:
        derived_from["dct:title"] = original_title
    if original_description is not None:
        derived_from["dct:type"] = original_description
    dataset["prov:wasDerivedFrom"] = derived_from

    jsonld.replace_distribution(dataset, vocab.coco_distribution(dataset_iri, coco_access_url))

    jsonld.ensure_context_prefixes(cloned, vocab.OUTPUT_CONTEXT_PREFIXES)
    cloned["@graph"] = [dataset]
    return cloned


def _augmented_title(original_title: str | None, segmentation_type: SegmentationType) -> str:
    base = original_title or "Untitled Dataset"
    return f"{base} — {segmentation_type.pretty} Segmentation"


def _augmented_description(
    original_description: str | None, segmentation_type: SegmentationType,
) -> str:
    if original_description:
        return original_description
    return (
        f"{segmentation_type.pretty} segmentation annotations produced by "
        f"Timeworx.io from client-provided source data."
    )
