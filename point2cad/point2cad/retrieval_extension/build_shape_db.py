"""Train the shape encoder, build the shape vector DB, and evaluate retrieval.

By default the encoder is trained only on plane/sphere/cylinder/cone, while the
DB also stores torus examples, so the torus row of the report shows how well a
type that the encoder never saw is recognized through retrieval alone.

    python build_shape_db.py --out artifacts
"""
import argparse
import os
import time

import numpy as np

from shape_elements import shape_descriptor
from shape_encoder import ShapeEncoder
from shape_rag import ShapeVectorDB, infer_shape_type
from shape_synth import make_dataset


def descriptors(patches):
    return np.stack([shape_descriptor(p) for p in patches])


def report(true, pred, types):
    print("true \\ pred " + " ".join(f"{t[:6]:>7s}" for t in types) + "   acc")
    for t in types:
        row = [np.sum((true == t) & (pred == p)) for p in types]
        n = max(np.sum(true == t), 1)
        print(f"{t:11s} " + " ".join(f"{c:7d}" for c in row) + f"  {row[types.index(t)] / n:.2f}")
    print(f"overall accuracy {np.mean(true == pred):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_types", nargs="+", default=["plane", "sphere", "cylinder", "cone"])
    ap.add_argument("--db_types", nargs="+", default=["plane", "sphere", "cylinder", "cone", "torus"])
    ap.add_argument("--n_train", type=int, default=300, help="training patches per type")
    ap.add_argument("--n_db", type=int, default=200, help="DB example patches per type")
    ap.add_argument("--n_test", type=int, default=100, help="test patches per type")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    train_p, train_y = make_dataset(args.train_types, args.n_train, seed=args.seed)
    db_p, db_y = make_dataset(args.db_types, args.n_db, seed=args.seed + 1)
    test_p, test_y = make_dataset(args.db_types, args.n_test, seed=args.seed + 2)
    Xtr, Xdb, Xte = descriptors(train_p), descriptors(db_p), descriptors(test_p)
    print(f"descriptors: train {Xtr.shape}, db {Xdb.shape}, test {Xte.shape} ({time.time() - t0:.0f}s)")

    enc = ShapeEncoder(Xtr.shape[1], seed=args.seed)
    enc.fit(Xtr, train_y, epochs=args.epochs, seed=args.seed, verbose=True)

    db = ShapeVectorDB()
    for z, t in zip(enc.embed_descriptors(Xdb), db_y):
        db.add(z, str(t))
    enc.save(os.path.join(args.out, "shape_encoder.npz"))
    db.save(os.path.join(args.out, "shape_db.npz"))

    Zte = enc.embed_descriptors(Xte)
    pred = np.array([next(iter(db.vote(z, args.k))) for z in Zte])
    unseen = sorted(set(args.db_types) - set(args.train_types))
    print(f"\n[retrieval with shape embedding]  (types not seen in training: {unseen or 'none'})")
    report(test_y, pred, list(args.db_types))

    # reference: same kNN vote on the raw standardized descriptor, no learning
    raw = ShapeVectorDB()
    nz = lambda X: ((X - enc.mu) / enc.sigma) / np.linalg.norm((X - enc.mu) / enc.sigma, axis=1, keepdims=True)
    for z, t in zip(nz(Xdb), db_y):
        raw.add(z, str(t))
    pred_raw = np.array([next(iter(raw.vote(z, args.k))) for z in nz(Xte)])
    print("\n[reference: retrieval on raw descriptor, no encoder]")
    report(test_y, pred_raw, list(args.db_types))

    # end-to-end API check on one test patch
    r = infer_shape_type(test_p[0], enc, db, k=args.k)
    print(f"\ninfer_shape_type(test[0]) true={test_y[0]} -> {r['type']} ({r['confidence']:.2f}), candidates={r['candidates']}")
    print(f"saved to {args.out}/ ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
