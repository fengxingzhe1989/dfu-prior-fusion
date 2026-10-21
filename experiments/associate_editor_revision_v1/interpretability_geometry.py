"""Pure, prediction-blind helpers for frozen interpretability analyses.

Only manifest identifiers, labels, groups and folds enter case selection.  All
random streams use an explicit seed plus a SHA-256 namespace; Python's unstable
process-local ``hash`` is never used.  These helpers neither read medical images
nor load models, priors or predictions.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd


INTERPRETABILITY_SEED = 20261002
CASE_COLUMNS = ["image", "label", "similarity_group", "fold"]


def _rng(seed: int, namespace: str, *identifiers: object) -> np.random.Generator:
    """Derive a stream independent of processing order and Python hash seed."""
    payload = json.dumps(
        [int(seed), namespace, *[str(item) for item in identifiers]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    entropy = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return np.random.default_rng(entropy)


def _binary(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("mask must be a nonempty two-dimensional array")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("mask must be binary (0/1 or bool), not a soft prior")
    return values.astype(bool, copy=False)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return occupied bounds ``(y_min, x_min, y_max, x_max)`` (inclusive).

    Empty masks return ``None``.  A full-width mask need not be immovable: its
    vertical margins still permit translations.
    """
    yy, xx = np.nonzero(_binary(mask))
    if not len(yy):
        return None
    return int(yy.min()), int(xx.min()), int(yy.max()), int(xx.max())


def mask_status(mask: np.ndarray) -> str:
    """Classify a mask as ``empty``, ``immovable``, or ``movable``."""
    binary = _binary(mask)
    bounds = mask_bbox(binary)
    if bounds is None:
        return "empty"
    y0, x0, y1, x1 = bounds
    height, width = binary.shape
    if y0 == x0 == 0 and y1 == height - 1 and x1 == width - 1:
        return "immovable"
    return "movable"


def translations(
    mask: np.ndarray,
    max_placements: int = 30,
    seed: int = INTERPRETABILITY_SEED,
    image: object = "",
) -> list[tuple[int, int]]:
    """Sample legal nonidentity ``(dy, dx)`` translations without replacement.

    Every occupied pixel remains in the image; the mask shape and area are
    preserved.  Up to ``max_placements`` positions are sampled with an
    image-specific fixed stream and returned in lexicographic order.  If fewer
    legal positions exist, return all of them.  Empty and immovable masks both
    return ``[]``; callers must use :func:`mask_status` to report them separately.
    """
    if not isinstance(max_placements, Integral) or max_placements < 1:
        raise ValueError("max_placements must be a positive integer")
    binary = _binary(mask)
    bounds = mask_bbox(binary)
    if bounds is None:
        return []
    y0, x0, y1, x1 = bounds
    height, width = binary.shape
    positions = [
        (dy, dx)
        for dy in range(-y0, height - y1)
        for dx in range(-x0, width - x1)
        if (dy, dx) != (0, 0)
    ]
    if len(positions) <= max_placements:
        return positions
    indices = _rng(seed, "mask_translation", image).choice(
        len(positions), size=int(max_placements), replace=False
    )
    return sorted(positions[int(index)] for index in indices)


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an entire binary mask with zero padding, never wrapping/cropping.

    Illegal translations raise ``ValueError`` rather than silently changing mask
    area.  The returned mask retains the input dtype and shape.  Translating an
    empty mask returns an empty mask (callers must not treat it as a control).
    """
    if not isinstance(dy, Integral) or not isinstance(dx, Integral):
        raise ValueError("dy and dx must be integers")
    values = np.asarray(mask)
    binary = _binary(values)
    bounds = mask_bbox(binary)
    result = np.zeros_like(values)
    if bounds is None:
        return result
    y0, x0, y1, x1 = bounds
    height, width = values.shape
    if y0 + dy < 0 or x0 + dx < 0 or y1 + dy >= height or x1 + dx >= width:
        raise ValueError("translation would move occupied pixels outside the image")
    yy, xx = np.nonzero(binary)
    result[yy + dy, xx + dx] = values[yy, xx]
    return result


def select_cases(
    frame: pd.DataFrame,
    seed: int = INTERPRETABILITY_SEED,
    per_class: int = 5,
) -> pd.DataFrame:
    """Select class-stratified cases without inspecting model performance.

    Required columns: ``image, label, similarity_group, fold``.  Additional
    columns (including predictions/scores) are ignored.  Return
    ``case_rank, image, label, similarity_group, fold``.  ``case_rank`` starts at
    one and is global across classes.  Input row order does not affect selection.

    Each class gets ``per_class`` distinct groups, and a group is never reused
    across classes (including mixed-label groups).  Seeded group preference
    lists plus a bipartite augmenting-path matching enforce these constraints
    without outcome-based retries; this is a randomized group-stratified design,
    not a claim of uniform sampling over all possible 20-case sets.  One image
    of the designated class is sampled within each selected group.  If the
    class/group constraints cannot be jointly satisfied, raise ``ValueError``
    rather than reduce the quota or try new seeds.
    """
    if not isinstance(per_class, Integral) or per_class < 1:
        raise ValueError("per_class must be a positive integer")
    missing = set(CASE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing case-selection columns: {sorted(missing)}")
    data = frame.loc[:, CASE_COLUMNS].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("case metadata must be nonempty and contain no missing values")
    data["image"] = data["image"].astype(str)
    data["similarity_group"] = data["similarity_group"].astype(str)
    if data["image"].duplicated().any():
        raise ValueError("each image must occur once in the input manifest")
    if (data.groupby("similarity_group")["fold"].nunique() > 1).any():
        raise ValueError("a similarity group must not span multiple outer folds")
    data = data.sort_values(CASE_COLUMNS, kind="stable").reset_index(drop=True)
    labels = sorted(data["label"].unique().tolist())
    preferences: dict[tuple[int, int], list[str]] = {}
    for label_index, label in enumerate(labels):
        groups = sorted(data.loc[data["label"] == label, "similarity_group"].unique())
        if len(groups) < per_class:
            raise ValueError(f"class {label!r} has fewer than {per_class} distinct groups")
        for slot in range(per_class):
            order = _rng(seed, "case_group", label, slot).permutation(len(groups))
            preferences[(label_index, slot)] = [groups[int(index)] for index in order]

    owner: dict[str, tuple[int, int]] = {}

    def assign(slot: tuple[int, int], seen: set[str]) -> bool:
        for group in preferences[slot]:
            if group in seen:
                continue
            seen.add(group)
            if group not in owner or assign(owner[group], seen):
                owner[group] = slot
                return True
        return False

    slots = list(preferences)
    for slot in slots:
        if not assign(slot, set()):
            raise ValueError("cannot meet all class quotas without reusing similarity groups")
    selected_by_slot = {slot: group for group, slot in owner.items()}
    rows = []
    for rank, slot in enumerate(slots, start=1):
        label = labels[slot[0]]
        group = selected_by_slot[slot]
        candidates = data.loc[
            (data["label"] == label) & (data["similarity_group"] == group)
        ].sort_values("image", kind="stable")
        index = int(_rng(seed, "case_image", label, group).integers(len(candidates)))
        row = candidates.iloc[index].to_dict()
        rows.append({"case_rank": rank, **row})
    result = pd.DataFrame(rows, columns=["case_rank", *CASE_COLUMNS])
    if result["similarity_group"].duplicated().any():
        raise AssertionError("internal error: selected groups are not globally unique")
    return result
