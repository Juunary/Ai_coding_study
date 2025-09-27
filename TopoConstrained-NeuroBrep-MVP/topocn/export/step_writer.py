
from __future__ import annotations
from typing import Optional
import os

def write_step(shape, out_path: str) -> bool:
    """Write OCCT shape to STEP if possible; return success flag."""
    try:
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone
    except Exception as e:
        print(f"[export] STEP export unavailable: {e}")
        return False
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(out_path)
    return bool(status==1)  # IFSelect_RetDone==1

def fallback_mesh_export(points, faces, out_obj: str):
    with open(out_obj, "w") as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
