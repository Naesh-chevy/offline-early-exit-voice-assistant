"""Label maps for FSC (action, object, location). Saved as json so train and device agree."""
import json


def build_label_map(values) -> dict:
    """Sorted unique values -> index. Sorted so the map is the same every run."""
    return {v: i for i, v in enumerate(sorted(set(values)))}


def save_label_maps(maps: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(maps, f, indent=2, sort_keys=True)


def load_label_maps(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def invert(label_map: dict) -> dict:
    return {i: v for v, i in label_map.items()}
