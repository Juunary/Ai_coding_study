
from __future__ import annotations
import argparse, os
from topocn.eval.bench_runner import run_impeller

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_impeller(args.data_dir, args.config, args.out_dir)

if __name__=="__main__":
    main()
