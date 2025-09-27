# make_labels_from_xyzc.py
# Usage:
#   python make_labels_from_xyzc.py <input_xyzc_file> <out_dir>
# Example:
#   python make_labels_from_xyzc.py impeller.xyzc ./impeller_numpy

import os
import sys
import numpy as np

# ---------- USER MAPPING (from your message) ----------
BODY_LAYERS = [1, 2, 3, 4, 5, 22]
WING1_LAYERS = [6, 12, 20, 22, 26]
WING2_LAYERS = [7, 13, 14, 15, 25]
WING3_LAYERS = [8, 10, 16, 17, 24]
WING4_LAYERS = [9, 11, 18, 19, 23]

# Precedence: Body > Wing1 > Wing2 > Wing3 > Wing4
GROUPS = [
    ("Body", BODY_LAYERS),
    ("Wing1", WING1_LAYERS),
    ("Wing2", WING2_LAYERS),
    ("Wing3", WING3_LAYERS),
    ("Wing4", WING4_LAYERS),
]

# Default types per segment (0:plane,1:cylinder,2:sphere,3:cone,4:inr)
# I choose Body -> cylinder (1), Wings -> INR (4) (good for thin/complex blades).
DEFAULT_TYPES = {
    "Body": 1,
    "Wing1": 4,
    "Wing2": 4,
    "Wing3": 4,
    "Wing4": 4,
}

def build_layer_mapping(groups):
    layer_to_group = {}
    conflicts = {}
    for grp_idx, (grp_name, layers) in enumerate(groups):
        for L in layers:
            if L in layer_to_group:
                # record conflict: existing owner vs new owner
                prev = layer_to_group[L]
                # keep previous (because groups is ordered by precedence with Body first)
                conflicts.setdefault(L, (prev, grp_name))
            else:
                layer_to_group[L] = grp_name
    return layer_to_group, conflicts

def main(input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading input:", input_path)
    # load numeric data: try numpy.loadtxt (space/comma delimited) or np.fromfile if binary etc.
    data = None
    try:
        data = np.loadtxt(input_path)
    except Exception as e:
        # try csv with comma
        try:
            data = np.loadtxt(input_path, delimiter=",")
        except Exception as e2:
            raise RuntimeError("Failed to load input file with numpy.loadtxt; please provide whitespace or CSV numeric file.") from e2

    if data.ndim == 1 and data.size % 4 == 0:
        data = data.reshape(-1, 4)
    if data.shape[1] < 4:
        raise ValueError("Input must have at least 4 columns: X Y Z C (C = layer/index). Got shape: " + str(data.shape))

    pts = data[:, :3].astype(np.float32)
    Ccol = data[:, 3]
    # Round C values to nearest integer (in case they are floats)
    Cints = np.rint(Ccol).astype(np.int64)

    # build mapping
    layer_to_group, conflicts = build_layer_mapping(GROUPS)
    if conflicts:
        print("WARNING: Conflicts detected for layers assigned to multiple groups. Resolved using precedence Body>Wing1>Wing2>Wing3>Wing4.")
        for L, (prev, new) in conflicts.items():
            print(f" - Layer {L} appeared in {prev} and {new}; assigned to {prev}")

    # Prepare group->segment id mapping (Body=0, Wing1=1, Wing2=2, Wing3=3, Wing4=4)
    group_order = [g[0] for g in GROUPS]
    segid_by_group = {grp: idx for idx, grp in enumerate(group_order)}
    # types array initial
    types_list = [DEFAULT_TYPES.get(grp, 4) for grp in group_order]

    # Map C ints to seg ids. If an encountered layer value is not in mapping, create a new extra segment for it.
    layer_to_segid = {}
    for L, grp in layer_to_group.items():
        layer_to_segid[L] = segid_by_group[grp]

    # Identify unique C values present in data
    unique_layers = np.unique(Cints)
    unmapped = [int(l) for l in unique_layers if int(l) not in layer_to_segid]
    extra_segments = []
    if unmapped:
        print("Found layers in the data not present in your mapping:", unmapped)
        print("They will be assigned as extra segments (type=INR=4) after the main 5 segments.")
        # create additional seg ids
        next_id = max(segid_by_group.values()) + 1
        for L in unmapped:
            layer_to_segid[L] = next_id
            extra_segments.append((L, next_id))
            types_list.append(4)  # INR default for extras
            next_id += 1

    # Now create labels array
    N = pts.shape[0]
    labels = np.empty(N, dtype=np.int64)
    for i in range(N):
        L = int(Cints[i])
        labels[i] = layer_to_segid.get(L, -1)  # should not be -1 after unmapped handling

    K = len(types_list)
    types_arr = np.array(types_list, dtype=np.int64)

    # Save outputs
    np.save(os.path.join(out_dir, "points.npy"), pts)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "types.npy"), types_arr)

    # Reporting
    print("Saved points.npy labels.npy types.npy to:", out_dir)
    print("points.npy shape:", pts.shape, "dtype:", pts.dtype)
    print("labels.npy shape:", labels.shape, "unique segments:", np.unique(labels))
    print("types.npy shape:", types_arr.shape, "types:", types_arr.tolist())
    print("Segment legend (id -> name):")
    for grp, idx in segid_by_group.items():
        print(f"  {idx} -> {grp}")
    if extra_segments:
        print("Extra segments created for unmapped layers:")
        for L, sid in extra_segments:
            print(f"  layer {L} -> seg {sid}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_labels_from_xyzc.py <input_xyzc.txt|.ply-converted> <out_dir>")
        sys.exit(1)
    in_path = sys.argv[1]
    outd = sys.argv[2]
    main(in_path, outd)
