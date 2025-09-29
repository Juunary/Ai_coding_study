# scripts/dump_edge_metrics.py  (requires graph/patches objects saved after run)
import pickle, json
graph = pickle.load(open("outputs/graph.pkl","rb"))
with open("outputs/edge_metrics.csv","w") as f:
    f.write("i,j,EG1_mean,EG2_mean,gap_mean,gap_count\n")
    for (i,j), ed in graph.edges.items():
        f.write(f"{i},{j},{ed.EG1:.6f},{ed.EG2:.6f},{ed.gap_mean:.6f},{ed.gap_count}\n")
