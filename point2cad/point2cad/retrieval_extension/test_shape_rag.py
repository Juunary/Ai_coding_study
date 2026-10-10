import numpy as np

from shape_elements import point_elements, shape_descriptor
from shape_encoder import ShapeEncoder
from shape_rag import ShapeVectorDB, infer_shape_type
from shape_synth import make_dataset, sample_patch


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def test_elements_match_geometry():
    rng = np.random.default_rng(0)
    sphere = point_elements(sample_patch("sphere", rng, noise_range=(0, 0), bleed_prob=0))
    cylinder = point_elements(sample_patch("cylinder", rng, noise_range=(0, 0), bleed_prob=0))
    assert np.median(sphere["shape_index"]) > 0.95 and np.median(sphere["gauss_sign"]) > 0.95
    assert abs(np.median(cylinder["shape_index"]) - 0.5) < 0.05
    assert abs(np.median(cylinder["gauss_sign"])) < 0.05
    assert cylinder["normal_tensor_eig"][0] < 0.01          # all normals orthogonal to the axis


def test_descriptor_pose_and_scale_invariant():
    rng = np.random.default_rng(1)
    p = sample_patch("cone", rng, n_range=(400, 401))
    q = 3.7 * p @ _random_rotation(rng).T + np.array([5.0, -2.0, 9.0])
    assert np.abs(shape_descriptor(p) - shape_descriptor(q)).max() < 0.02


def test_triplet_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 10))
    y = np.repeat(np.array(["a", "b", "c"]), 8)
    enc = ShapeEncoder(10, hidden=8, out_dim=4, seed=1)
    _, g = enc._triplet_step(X, y, 0.5)
    P = enc.params["W1"]
    old = P[3, 2]
    P[3, 2] = old + 1e-6
    lp, _ = enc._triplet_step(X, y, 0.5)
    P[3, 2] = old - 1e-6
    lm, _ = enc._triplet_step(X, y, 0.5)
    P[3, 2] = old
    assert abs((lp - lm) / 2e-6 - g["W1"][3, 2]) < 1e-5


def test_unseen_type_found_by_retrieval():
    train_p, train_y = make_dataset(["plane", "sphere", "cylinder", "cone"], 40, seed=0)
    enc = ShapeEncoder(len(shape_descriptor(train_p[0])), seed=0)
    enc.fit(np.stack([shape_descriptor(p) for p in train_p]), train_y, epochs=60)

    db = ShapeVectorDB()
    db_p, db_y = make_dataset(["plane", "sphere", "cylinder", "cone", "torus"], 30, seed=1)
    for p, t in zip(db_p, db_y):                                 # torus only in the DB
        db.add(enc.embed(p), str(t))

    test_p, test_y = make_dataset(["torus", "sphere"], 15, seed=2)
    pred = np.array([infer_shape_type(p, enc, db)["type"] for p in test_p])
    assert np.mean(pred[test_y == "torus"] == "torus") >= 0.8
    assert np.mean(pred[test_y == "sphere"] == "sphere") >= 0.8


def test_save_load_and_fitting_verification():
    import os
    import tempfile

    rng = np.random.default_rng(3)
    enc = ShapeEncoder(len(shape_descriptor(sample_patch("plane", rng))), seed=0)
    db = ShapeVectorDB()
    db.add_patches(enc, [sample_patch("plane", rng) for _ in range(5)], "plane")
    db.add_patches(enc, [sample_patch("sphere", rng) for _ in range(5)], "sphere")
    with tempfile.TemporaryDirectory() as d:
        enc.save(os.path.join(d, "e.npz"))
        db.save(os.path.join(d, "d.npz"))
        enc2 = ShapeEncoder.load(os.path.join(d, "e.npz"))
        db2 = ShapeVectorDB.load(os.path.join(d, "d.npz"))
    p = sample_patch("sphere", rng)
    assert np.allclose(enc.embed(p), enc2.embed(p))
    assert db2.labels == db.labels

    # retrieved candidates restrict the error-based selection
    errors = {"plane": 0.001, "sphere": 0.002, "cylinder": 0.5, "cone": np.inf, "inr": 0.3}
    r = infer_shape_type(p, enc2, db2, errors=errors)
    assert r["type"] in r["candidates"] | {"inr"}


if __name__ == "__main__":
    test_elements_match_geometry()
    test_descriptor_pose_and_scale_invariant()
    test_triplet_gradient()
    test_unseen_type_found_by_retrieval()
    test_save_load_and_fitting_verification()
    print("ok")
