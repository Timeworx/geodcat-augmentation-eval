import copy


def inject_semantic_uris(coco: dict, class_to_agrovoc: dict[str, str]) -> dict:
    """Return a copy of `coco` with `semantic_uri` added to each category whose
    `name` is present in `class_to_agrovoc`.

    Standard COCO consumers still parse the result; we only ADD a field.
    Unknown categories (not in the map) are left untouched.
    """
    if not isinstance(coco, dict):
        raise TypeError("coco must be a dict")
    result = copy.deepcopy(coco)
    categories = result.get("categories")
    if not isinstance(categories, list):
        return result
    for category in categories:
        if not isinstance(category, dict):
            continue
        name = category.get("name")
        if isinstance(name, str) and name in class_to_agrovoc:
            category["semantic_uri"] = class_to_agrovoc[name]
    return result
