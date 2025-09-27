# convert_labels.py
import numpy as np
import os

# Suppose you have a CSV with last column = label, or a scalar file
data = np.loadtxt("points_with_labels.csv", delimiter=",")  # or load your format
points = data[:, :3].astype(np.float32)
labels = data[:, 3].astype(np.int64)

np.save("points.npy", points)
np.save("labels.npy", labels)
print("Saved points.npy, labels.npy", points.shape, labels.shape)
