from src.metadata import jsonld
from src.metadata.augmenter import augment_ai_ready
from src.metadata.types import SegmentationType
from src.metadata.vocab import TIMEWORX_DATASET_NS

JOB_ID = "1b4102eb-2a11-4cb2-b01a-a8696ba43770"
COCO_URL = "https://drive.google.com/file/d/187ng7pWGsEhqJWg3xWVJE2WmZdYNwLNh/view?usp=sharing"


def _augment(input_geodcat: dict, seg_type: SegmentationType = SegmentationType.INSTANCE) -> dict:
    return augment_ai_ready(
        input_geodcat,
        job_id=JOB_ID,
        segmentation_type=seg_type,
        coco_access_url=COCO_URL,
    )


def test_input_is_not_mutated(input_geodcat: dict) -> None:
    before = jsonld.clone(input_geodcat)
    _augment(input_geodcat)
    assert input_geodcat == before


def test_top_level_identity_fields_replaced(input_geodcat: dict) -> None:
    out = _augment(input_geodcat)
    node = jsonld.find_dataset(out)
    assert node["@id"] == f"{TIMEWORX_DATASET_NS}/{JOB_ID}"
    assert node["dct:identifier"] == JOB_ID
    assert node["dct:publisher"]["@id"] == "https://timeworx.io/organization"
    assert node["dct:publisher"]["foaf:name"] == "Timeworx.io"


def test_title_follows_pseudocode_rule(input_geodcat: dict) -> None:
    out = _augment(input_geodcat, SegmentationType.INSTANCE)
    node = jsonld.find_dataset(out)
    assert node["dct:title"] == (
        "AgrifoodTEF: Time series of maize images (RGB) — Instance Segmentation"
    )


def test_zero_loss_for_unreplaced_predicates(input_geodcat: dict) -> None:
    """Every input predicate that isn't on the explicit-replace list must
    survive on the output dataset node. dcat:keyword is modified additively
    — every input keyword still appears."""
    input_node = jsonld.find_dataset(input_geodcat)
    out = _augment(input_geodcat)
    out_node = jsonld.find_dataset(out)
    replaced = {"@id", "dct:title", "dct:identifier", "dct:publisher", "dcat:distribution"}
    additive = {"dcat:keyword"}
    for key, value in input_node.items():
        if key in replaced:
            continue
        assert key in out_node, f"input predicate missing on output: {key}"
        if key in additive:
            # Original entries must survive; additions are allowed.
            for entry in value:
                assert entry in out_node[key], f"input keyword dropped: {entry}"
            continue
        assert out_node[key] == value, f"input predicate altered on output: {key}"


def test_keywords_union_no_duplicates(input_geodcat: dict) -> None:
    out = _augment(input_geodcat, SegmentationType.INSTANCE)
    node = jsonld.find_dataset(out)
    values = [kw["@value"] for kw in node["dcat:keyword"]]
    assert values == [
        "Phenotyping",
        "Precision Agriculture",
        "Time series images of maize plants from one field",
        "Instance Segmentation",
        "Plant Counting",
    ]


def test_processing_keywords_per_type(input_geodcat: dict) -> None:
    semantic = _augment(input_geodcat, SegmentationType.SEMANTIC)
    panoptic = _augment(input_geodcat, SegmentationType.PANOPTIC)
    sem_values = [kw["@value"] for kw in jsonld.find_dataset(semantic)["dcat:keyword"]]
    pan_values = [kw["@value"] for kw in jsonld.find_dataset(panoptic)["dcat:keyword"]]
    assert "Semantic Segmentation" in sem_values
    assert "Instance Segmentation" not in sem_values
    assert "Panoptic Segmentation" in pan_values
    assert "Advanced Phenotyping" in pan_values


def test_lineage_points_at_input_id(input_geodcat: dict) -> None:
    original_id = jsonld.find_dataset(input_geodcat)["@id"]
    out = _augment(input_geodcat)
    derived = jsonld.find_dataset(out)["prov:wasDerivedFrom"]
    assert derived["@id"] == original_id
    assert derived["dct:title"] == "AgrifoodTEF: Time series of maize images (RGB)"


def test_distribution_replaced_with_coco(input_geodcat: dict) -> None:
    out = _augment(input_geodcat)
    dists = jsonld.find_dataset(out)["dcat:distribution"]
    assert len(dists) == 1
    coco = dists[0]
    assert coco["@id"] == f"{TIMEWORX_DATASET_NS}/{JOB_ID}#dist-coco"
    assert coco["dct:title"] == "Instance Annotations (COCO Format)"
    assert coco["dcat:accessURL"]["@id"] == COCO_URL
    assert coco["dct:conformsTo"]["dct:title"] == "MS COCO Object Detection Format"


def test_context_extended_with_prov(input_geodcat: dict) -> None:
    out = _augment(input_geodcat)
    assert out["@context"]["prov"] == "http://www.w3.org/ns/prov#"
    assert out["@context"]["spdx"] == "http://spdx.org/rdf/terms#"
    assert out["@context"]["locn"] == "http://www.w3.org/ns/locn#"


def test_matches_fixture_structure(input_geodcat: dict, output_geodcat: dict) -> None:
    """The augmented document and the canonical output fixture have the same
    top-level shape and the same set of dataset predicates (modulo the
    title/description that the fixture domain-edits)."""
    out = _augment(input_geodcat)
    assert set(out.keys()) == set(output_geodcat.keys())
    out_node = jsonld.find_dataset(out)
    fixture_node = jsonld.find_dataset(output_geodcat)
    assert set(out_node.keys()) == set(fixture_node.keys())
