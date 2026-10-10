import numpy as np
from select_type import select_surface_type
from retrieval_candidates import SupportSet


def test_matches_original_rules():
    # basic primitive wins when inr is only slightly better
    assert select_surface_type({"plane": 0.02, "sphere": 0.05, "cylinder": 0.03, "cone": np.inf, "inr": 0.0195}) == "plane"
    # inr wins when clearly better
    assert select_surface_type({"plane": 0.05, "sphere": 0.06, "cylinder": 0.07, "cone": np.inf, "inr": 0.01}) == "inr"
    # small basic error is always preferred over inr
    assert select_surface_type({"plane": 0.005, "sphere": 0.05, "cylinder": 0.07, "cone": np.inf, "inr": 0.001}) == "plane"
    # cone ~ plane collapses to plane
    assert select_surface_type({"plane": 0.01, "sphere": 0.05, "cylinder": 0.07, "cone": 0.010001, "inr": 0.5}) == "plane"


def test_new_type_and_candidate_mask():
    errs = {"plane": 0.1, "sphere": 0.1, "cylinder": 0.1, "cone": np.inf, "torus": 0.004, "inr": 0.5}
    assert select_surface_type(errs) == "torus"
    assert select_surface_type(errs, candidates={"cylinder"}) == "cylinder"


def test_retrieval():
    s = SupportSet()
    s.add(np.random.RandomState(0).normal(0, 0.1, (20, 8)), "plane")
    s.add(np.random.RandomState(1).normal(5, 0.1, (20, 8)), "torus")
    assert s.candidates(np.full(8, 5.0), k=5) == {"torus"}


if __name__ == "__main__":
    test_matches_original_rules()
    test_new_type_and_candidate_mask()
    test_retrieval()
    print("ok")
