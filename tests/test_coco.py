from src.metadata.coco import inject_semantic_uris


def test_injects_semantic_uri_on_matching_category() -> None:
    coco = {
        "categories": [
            {"id": 1, "name": "maize plant", "supercategory": "plant"},
        ],
        "annotations": [],
    }
    out = inject_semantic_uris(coco, {"maize plant": "http://aims.fao.org/aos/agrovoc/c_12332"})
    assert out["categories"][0]["semantic_uri"] == "http://aims.fao.org/aos/agrovoc/c_12332"


def test_does_not_mutate_input() -> None:
    coco = {"categories": [{"id": 1, "name": "x"}]}
    inject_semantic_uris(coco, {"x": "http://example/"})
    assert "semantic_uri" not in coco["categories"][0]


def test_leaves_unmapped_categories_untouched() -> None:
    coco = {
        "categories": [
            {"id": 1, "name": "x"},
            {"id": 2, "name": "y"},
        ]
    }
    out = inject_semantic_uris(coco, {"x": "http://example/x"})
    assert out["categories"][0]["semantic_uri"] == "http://example/x"
    assert "semantic_uri" not in out["categories"][1]


def test_preserves_existing_coco_fields(sample_coco: dict) -> None:
    """Standard COCO consumers still parse the result; we only ADD a field."""
    # Drop the fixture's pre-existing semantic_uri so we can re-inject.
    for cat in sample_coco["categories"]:
        cat.pop("semantic_uri", None)
    out = inject_semantic_uris(sample_coco, {"maize plant": "http://aims.fao.org/aos/agrovoc/c_12332"})
    assert out["info"] == sample_coco["info"]
    assert out["images"] == sample_coco["images"]
    assert out["annotations"] == sample_coco["annotations"]
    # And the new field is added
    assert out["categories"][0]["semantic_uri"] == "http://aims.fao.org/aos/agrovoc/c_12332"


def test_handles_missing_categories_gracefully() -> None:
    coco = {"info": {}, "images": [], "annotations": []}
    out = inject_semantic_uris(coco, {"x": "http://example/"})
    assert out == coco
