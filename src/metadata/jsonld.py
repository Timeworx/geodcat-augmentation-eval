import copy
from typing import Any


def find_dataset(graph_doc: dict) -> dict:
    """Locate the dcat:Dataset node inside the @graph array.

    Returns the *same* dict reference (no copy) — callers that mutate should
    clone first. Raises ValueError if no dataset node is found.
    """
    nodes = graph_doc.get("@graph", [])
    if not isinstance(nodes, list):
        raise ValueError("@graph must be a list")
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("@type")
        if _is_dataset_type(node_type):
            return node
    raise ValueError("no dcat:Dataset node found in @graph")


def _is_dataset_type(t: Any) -> bool:
    if t == "dcat:Dataset":
        return True
    if isinstance(t, list) and "dcat:Dataset" in t:
        return True
    return False


def clone(node: dict) -> dict:
    return copy.deepcopy(node)


def append_keyword(node: dict, keyword: str | dict, *, language: str = "en") -> None:
    """Append a keyword to dcat:keyword. Skips duplicates (by @value or by string)."""
    keywords = node.setdefault("dcat:keyword", [])
    if not isinstance(keywords, list):
        # Single existing keyword promoted to a list
        keywords = [keywords]
        node["dcat:keyword"] = keywords
    new_entry: dict | str
    if isinstance(keyword, str):
        new_entry = {"@language": language, "@value": keyword}
    else:
        new_entry = keyword
    if _keyword_in(keywords, new_entry):
        return
    keywords.append(new_entry)


def _keyword_in(keywords: list, candidate: Any) -> bool:
    candidate_value = candidate.get("@value") if isinstance(candidate, dict) else candidate
    for existing in keywords:
        existing_value = existing.get("@value") if isinstance(existing, dict) else existing
        if existing_value == candidate_value:
            return True
    return False


def replace_distribution(node: dict, distribution: dict) -> None:
    """Replace dcat:distribution with a single new distribution entry."""
    node["dcat:distribution"] = [distribution]


def ensure_context_prefixes(graph_doc: dict, prefixes: dict[str, str]) -> None:
    """Add namespace prefixes to @context if missing. Mutates in place."""
    ctx = graph_doc.setdefault("@context", {})
    if not isinstance(ctx, dict):
        # @context may be a list or a string; only extend dict-shaped contexts.
        return
    for prefix, iri in prefixes.items():
        ctx.setdefault(prefix, iri)
