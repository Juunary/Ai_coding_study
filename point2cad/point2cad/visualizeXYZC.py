import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

def load_xyzc(path):
    data = np.loadtxt(path)
    if data.shape[1] != 4:
        raise ValueError("Input file must have 4 columns: x, y, z, label")
    return data

def print_layer_distribution(label_array):
    total = len(label_array)
    label_counts = Counter(label_array)
    print("📊 Layer distribution (by percentage):")
    for label, count in sorted(label_counts.items()):
        percent = 100.0 * count / total
        print(f"  Layer {label:.6f}: {count} points ({percent:.2f}%)")
    print("")

def visualize_point_cloud(data):
    x, y, z, label = data[:,0], data[:,1], data[:,2], data[:,3]
    print_layer_distribution(label)

    unique_labels = np.unique(label)

    # Normalize labels to color map
    label_to_color = {l: plt.cm.tab20(i / max(len(unique_labels)-1, 1)) 
                      for i, l in enumerate(unique_labels)}
    colors = np.array([label_to_color[l] for l in label])

    # 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization by Layer')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize .xyzc point cloud with color by layer.')
    parser.add_argument('--path_in', required=True, help='Path to .xyzc file')
    args = parser.parse_args()

    data = load_xyzc(args.path_in)
    visualize_point_cloud(data)

if __name__ == '__main__':
    main()
