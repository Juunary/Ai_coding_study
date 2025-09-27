# make_labels_dbscan.py
import numpy as np
from sklearn.cluster import DBSCAN

pts = np.load("points.npy")  # (N,3)
cl = DBSCAN(eps=0.01, min_samples=10).fit(pts)
labels = cl.labels_.astype(np.int64)  # -1 means noise
# convert noise (-1) to a new label if you want
labels[labels==-1] = labels.max()+1
np.save("labels.npy", labels)
print("Saved labels.npy unique labels:", np.unique(labels))
