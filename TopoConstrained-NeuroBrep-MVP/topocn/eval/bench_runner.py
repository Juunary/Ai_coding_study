from __future__ import annotations
import os, json, yaml, numpy as np, torch
from ..optimize.solver import solve, SolveConfig
from ..export.brep import assemble_brep
from ..export.step_writer import write_step
from ..eval.metrics_cad import count_g1_discontinuities, step_success
from ..eval.metrics_geom import chamfer, hausdorff
from ..topo.graph import Graph
from ..utils import log
from ..topo.validators import gap_violations, self_intersection_soft
import pickle, csv

def save_edge_metrics(graph, out_dir):
    with open(os.path.join(out_dir, 'edge_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['edge_id', 'i', 'j', 'EG1', 'EG2', 'gap_mean', 'gap_std', 'violations'])
        
        for (i, j), ed in graph.edges.items():
            EG1 = ed.EG1
            EG2 = ed.EG2
            gap_mean = ed.gap_mean
            gap_std = ed.gap_std
            violations = ed.gap_count  # 또는 다른 지표
            writer.writerow([f'{i}_{j}', i, j, EG1, EG2, gap_mean, gap_std, violations])

def run_impeller(data_dir: str, config_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    # 데이터 로드
    pts = np.load(os.path.join(data_dir, "points.npy")).astype(np.float32)
    labels = np.load(os.path.join(data_dir, "labels.npy")).astype(np.int64)
    types = np.load(os.path.join(data_dir, "types.npy")).astype(np.int64)

    # Solver 설정
    cfg = SolveConfig(**cfg_yaml.get("solver", {}))
    patches, graph, scal = solve(pts, labels, types, cfg)

    # STEP 파일로 내보내기
    shape = assemble_brep(patches, graph)
    step_path = os.path.join(out_dir, "result.step")
    step_ok = False
    if shape is not None:
        step_ok = write_step(shape, step_path)

    # G1 불연속성 계산
    edge_labels = {(i, j): ed.labels for (i, j), ed in graph.edges.items()}
    g1_disc = count_g1_discontinuities(edge_labels, tau_deg=5.0)

    # 기하학적 메트릭 (Chamfer/Hausdorff)
    P = torch.from_numpy(pts)
    Pfit = []
    for pid, patch in patches.items():
        Pi = P[labels == pid]
        if Pi.shape[0] == 0:
            continue
        if hasattr(patch, "project"):
            Pfit.append(patch.project(Pi))
        else:
            Pfit.append(Pi)
    if len(Pfit) > 0:
        Pfit = torch.cat(Pfit, dim=0)
    else:
        Pfit = P.clone()

    CD = chamfer(P, Pfit)
    HD = hausdorff(P, Pfit)

    # gap & self-soft 메트릭 집계
    gap_count = 0
    self_soft = 0.0
    for (i, j), ed in graph.edges.items():
        gap_count += gap_violations(ed.pairs_i, ed.pairs_j, eps=1e-3)
        self_soft += self_intersection_soft(ed.pairs_i, ed.pairs_j, tau=1e-3)

    # 최종 보고서 생성
    report = dict(
        E = scal,
        G1_discont = g1_disc,
        GAP_violations = int(gap_count),
        SELF_soft = float(self_soft),
        STEP_success = bool(step_ok),
        Chamfer = float(CD),
        Hausdorff = float(HD)
    )

    # 로그로 출력
    log.info(f"Report: {report}")

    # 보고서를 pickle 형식으로 저장
    pickle.dump(patches, open(os.path.join(out_dir,"patches.pkl"), "wb"))
    pickle.dump(graph, open(os.path.join(out_dir,"graph.pkl"), "wb"))
    # also save iteration history if available
    if hasattr(scal, "history"):
        json.dump(scal.history, open(os.path.join(out_dir,"history.json"), "w"))

    # 보고서를 'out_dir'에 저장 (수정된 경로)
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if shape is None:
        log.warn("STEP not produced (pythonocc-core missing). You can still inspect report.json.")
