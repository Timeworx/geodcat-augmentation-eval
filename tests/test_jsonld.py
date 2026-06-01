import pytest

from src.metadata import jsonld


def test_find_dataset_in_input(input_geodcat: dict) -> None:
    node = jsonld.find_dataset(input_geodcat)
    assert node["@id"].startswith("did:op:")
    assert node["@type"] == "dcat:Dataset"


def test_find_dataset_raises_when_absent() -> None:
    with pytest.raises(ValueError):
        jsonld.find_dataset({"@graph": [{"@type": "dcat:Distribution"}]})


def test_clone_is_deep(input_geodcat: dict) -> None:
    node = jsonld.find_dataset(input_geodcat)
    cloned = jsonld.clone(node)
    cloned["dct:title"] = "mutated"
    cloned["dcat:keyword"].append({"@language": "en", "@value": "extra"})
    assert jsonld.find_dataset(input_geodcat)["dct:title"] != "mutated"
    assert {"@language": "en", "@value": "extra"} not in jsonld.find_dataset(input_geodcat)["dcat:keyword"]


def test_append_keyword_dedupes(input_geodcat: dict) -> None:
    node = jsonld.clone(jsonld.find_dataset(input_geodcat))
    before = len(node["dcat:keyword"])
    jsonld.append_keyword(node, "Phenotyping")  # duplicate of existing
    jsonld.append_keyword(node, "Instance Segmentation")  # new
    assert len(node["dcat:keyword"]) == before + 1
    values = [kw["@value"] for kw in node["dcat:keyword"]]
    assert values.count("Phenotyping") == 1
    assert "Instance Segmentation" in values


def test_replace_distribution_replaces_existing(input_geodcat: dict) -> None:
    node = jsonld.clone(jsonld.find_dataset(input_geodcat))
    new_dist = {"@id": "x", "@type": "dcat:Distribution"}
    jsonld.replace_distribution(node, new_dist)
    assert node["dcat:distribution"] == [new_dist]


def test_ensure_context_prefixes_adds_missing(input_geodcat: dict) -> None:
    doc = jsonld.clone(input_geodcat)
    jsonld.ensure_context_prefixes(doc, {"prov": "http://www.w3.org/ns/prov#"})
    assert doc["@context"]["prov"] == "http://www.w3.org/ns/prov#"


def test_ensure_context_prefixes_preserves_existing() -> None:
    doc = {"@context": {"prov": "http://existing/"}}
    jsonld.ensure_context_prefixes(doc, {"prov": "http://other/"})
    assert doc["@context"]["prov"] == "http://existing/"
