"""Slim conftest for the evaluation framework.

Provides only the JSON-LD / COCO fixture loaders the kept unit tests
(`test_augmenter.py`, `test_skeleton.py`, `test_coco.py`, `test_jsonld.py`)
depend on. The sidecar-side fixtures (`broker_state`, `client`, FastAPI
TestClient, in-memory correlation store, stub services) were removed
along with the sidecar source they wrapped.
"""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def input_geodcat() -> dict:
    return _load("input-geodcat.jsonld")


@pytest.fixture
def output_geodcat() -> dict:
    return _load("output-geodcat.jsonld")


@pytest.fixture
def skeleton_geodcat() -> dict:
    return _load("skeleton-geodcat.jsonld")


@pytest.fixture
def sample_coco() -> dict:
    return _load("sample-coco.json")
