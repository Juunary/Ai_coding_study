
from __future__ import annotations
import numpy as np

def make_synthetic_impeller(n_blades:int=4, n_per_blade:int=500, noise:float=0.002):
    """Create a tiny synthetic 'impeller-like' point cloud with 3 segment types:
    hub cylinder (label 0, cylinder), shroud plane (label 1, plane), blades approximate planes (labels 2..).
    types: [cylinder, plane, plane, plane, ...]
    """
    pts=[]; labels=[]; types=[]
    # hub cylinder along z
    R=0.2; H=0.2
    for t in np.linspace(0,2*np.pi, n_per_blade):
        for z in np.linspace(-H, H, 10):
            x = R*np.cos(t); y=R*np.sin(t)
            pts.append([x,y,z])
    labels += [0]*len(pts)
    types.append(1)  # cylinder
    # shroud plane z=H
    n_plane = n_per_blade*5
    for _ in range(n_plane):
        x = np.random.uniform(-0.4,0.4); y=np.random.uniform(-0.4,0.4); z=H
        pts.append([x,y,z])
    labels += [1]*n_plane; types.append(0)  # plane
    # blades: n_blades vertical planes rotated
    start = len(pts)
    for b in range(n_blades):
        ang = b*(2*np.pi/n_blades)
        n = np.array([np.cos(ang), np.sin(ang), 0.0])
        for _ in range(n_per_blade):
            r = np.random.uniform(R, 0.35)
            z = np.random.uniform(-H, H)
            p0 = np.array([r,0,z]); # rotate p0 so that n·p=d
            # choose d=0 to pass through origin; generate 2D line along direction orth to n in xy-plane
            # construct orth basis
            tdir = np.array([-np.sin(ang), np.cos(ang), 0.0])
            p = r*n + z*np.array([0,0,1.0])
            pts.append(p.tolist())
        labels += [2+b]*n_per_blade
        types.append(0)  # plane
    pts = np.array(pts, dtype=np.float32)
    # add noise
    pts += noise*np.random.randn(*pts.shape).astype(np.float32)
    labels = np.array(labels[:len(pts)], dtype=np.int64)
    types = np.array(types, dtype=np.int64)
    return pts, labels, types
