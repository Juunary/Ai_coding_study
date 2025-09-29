# scripts/inspect_report.py
import json, sys, pathlib, pprint
p = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "outputs/report.json")
r = json.loads(p.read_text(encoding="utf-8"))
pprint.pprint(r)
# quick thresholds
E = r.get("E",{})
print("Etotal:", E.get("Etotal"))
print("GAP violations:", r.get("GAP_violations"))
print("G1 discontinuities:", r.get("G1_discont"))
