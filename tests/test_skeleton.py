from src.metadata import jsonld
from src.metadata.skeleton import build_legacy_skeleton
from src.metadata.types import SegmentationType


JOB_ID = "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv"
RAW_URI = "s3://client-external-bucket/raw-data/batch-01"
COCO_URL = "https://api.timeworx.io/download/a1b2c3d4-output.json"


def test_no_spatial_or_temporal() -> None:
    skel = build_legacy_skeleton(
        job_id=JOB_ID,
        segmentation_type=SegmentationType.INSTANCE,
        raw_uri=RAW_URI,
        target_class="cauliflower",
        coco_access_url=COCO_URL,
    )
    node = jsonld.find_dataset(skel)
    assert "dct:spatial" not in node
    assert "dct:temporal" not in node


def test_keywords_are_literal_strings() -> None:
    skel = build_legacy_skeleton(
        job_id=JOB_ID,
        segmentation_type=SegmentationType.INSTANCE,
        raw_uri=RAW_URI,
        target_class="cauliflower",
        coco_access_url=COCO_URL,
    )
    node = jsonld.find_dataset(skel)
    values = [kw["@value"] for kw in node["dcat:keyword"]]
    assert values == ["Instance Segmentation", "cauliflower", "Computer Vision"]


def test_provenance_points_at_raw_uri() -> None:
    skel = build_legacy_skeleton(
        job_id=JOB_ID,
        segmentation_type=SegmentationType.INSTANCE,
        raw_uri=RAW_URI,
        target_class="cauliflower",
        coco_access_url=COCO_URL,
    )
    derived = jsonld.find_dataset(skel)["prov:wasDerivedFrom"]
    assert derived["@id"] == RAW_URI
    assert derived["dct:description"] == "Client-provided raw imagery URI"


def test_matches_fixture_shape(skeleton_geodcat: dict) -> None:
    skel = build_legacy_skeleton(
        job_id=JOB_ID,
        segmentation_type=SegmentationType.INSTANCE,
        raw_uri=RAW_URI,
        target_class="cauliflower",
        coco_access_url=COCO_URL,
    )
    # Top-level keys
    assert set(skel.keys()) == set(skeleton_geodcat.keys())
    # Dataset-node keys
    assert set(jsonld.find_dataset(skel).keys()) == set(jsonld.find_dataset(skeleton_geodcat).keys())
