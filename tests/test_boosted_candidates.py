import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from run_boosted_candidates import dense_features

def test_feature_shape_and_finiteness():
    x = dense_features("Name: Ada Lovelace", (6, 18, "NAME"), .5)
    assert len(x) == 52
    assert all(v == v for v in x)

def test_feature_has_label_specific_numeric_prefix():
    assert dense_features("x", (0, 1, "EMAIL"), .25)[0] != dense_features("x", (0, 1, "NAME"), .25)[0]
