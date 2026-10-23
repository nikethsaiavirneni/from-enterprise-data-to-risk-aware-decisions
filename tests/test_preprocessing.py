import numpy as np
import pandas as pd

from preprocessing import assign_split, on_shelf_mask, sample_items, split_bounds


def test_split_is_contiguous_and_covers_all_days():
    b = split_bounds(1941, 91, 91)
    assert b["train"][0] == 1 and b["test"][1] == 1941
    assert b["train"][1] + 1 == b["val"][0]
    assert b["val"][1] + 1 == b["test"][0]
    assert b["val"][1] - b["val"][0] + 1 == 91
    assert b["test"][1] - b["test"][0] + 1 == 91


def test_assign_split_is_chronological_and_never_shuffles():
    b = split_bounds(100, 10, 10)
    s = assign_split(np.arange(1, 101), b)
    assert (np.diff(s) >= 0).all()                       # train -> val -> test only
    assert (s == 0).sum() == 80 and (s == 1).sum() == 10 and (s == 2).sum() == 10


def test_on_shelf_mask_drops_weeks_without_price():
    price_wide = np.array([[np.nan, 2.0, 3.0], [1.0, np.nan, 3.0]], dtype=np.float32)
    week_pos = np.array([0, 0, 1, 2])
    m = on_shelf_mask(price_wide, week_pos)
    assert m.tolist() == [[False, False, True, True], [True, True, False, True]]


def test_sample_items_is_reproducible_and_stratified():
    items = pd.DataFrame({"item_id": [f"i{k}" for k in range(60)],
                          "dept_id": ["A"] * 40 + ["B"] * 20})
    a = sample_items(items, 12, seed=1)
    assert a == sample_items(items, 12, seed=1)
    assert a != sample_items(items, 12, seed=2)
    depts = items.set_index("item_id").loc[a, "dept_id"]
    assert set(depts) == {"A", "B"} and (depts == "A").sum() == 8
