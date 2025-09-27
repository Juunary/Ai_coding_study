
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

def assemble_brep(patches: Dict[int, object], graph) -> Optional[object]:
    """Construct a BRep shape via OpenCascade if available. Returns OCCT TopoDS_Shape or None.
    This MVP attempts a very simple sewing of faces from primitives; trimming by boundary wires is heuristic.
    """
    try:
        from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax3, gp_Ax2, gp_Pln, gp_Ax1, gp_Circ
        from OCC.Core.Geom import Geom_Plane, Geom_CylindricalSurface, Geom_SphericalSurface, Geom_ConicalSurface
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_Sewing
        from OCC.Core.TopoDS import TopoDS_Shape
    except Exception as e:
        print(f"[export] pythonocc-core not available or failed to import: {e}")
        return None
    # TODO: Implement actual trimming with boundary wires; here we just create unbounded faces—validity not guaranteed.
    faces = []
    for pid, p in patches.items():
        if getattr(p, "kind", "")=="plane":
            n = p.n.detach().cpu().numpy(); d = float(p.d.detach().cpu().numpy())
            # Plane at distance along normal; approximate as face on a large square
            pln = gp_Pln(gp_Pnt(0,0,-d*n[2] if abs(n[2])>1e-6 else 0), gp_Dir(*n))
            F = BRepBuilderAPI_MakeFace(pln, -1000, 1000, -1000, 1000).Face()
            faces.append(F)
        elif getattr(p, "kind","")=="cylinder":
            a = p.a.detach().cpu().numpy(); c = p.c.detach().cpu().numpy(); r = abs(float(p.r.detach().cpu().numpy()))
            ax = gp_Ax3(gp_Pnt(*c), gp_Dir(*a))
            from OCC.Core.Geom import Geom_CylindricalSurface
            surf = Geom_CylindricalSurface(ax, r)
            F = BRepBuilderAPI_MakeFace(surf, -1000, 1000, -3.14, 3.14).Face()
            faces.append(F)
        elif getattr(p, "kind","")=="sphere":
            c = p.c.detach().cpu().numpy(); r = abs(float(p.r.detach().cpu().numpy()))
            from OCC.Core.Geom import Geom_SphericalSurface
            from OCC.Core.gp import gp_Sphere
            # Spherical surface face
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
            F = BRepPrimAPI_MakeSphere(gp_Pnt(*c), r).Shape()
            faces.append(F)
        elif getattr(p, "kind","")=="cone":
            a = p.a.detach().cpu().numpy(); v = p.vtx.detach().cpu().numpy(); ang = abs(float(p.ang.detach().cpu().numpy()))
            ax = gp_Ax3(gp_Pnt(*v), gp_Dir(*a))
            surf = Geom_ConicalSurface(ax, ang, 1.0)
            F = BRepBuilderAPI_MakeFace(surf, -1000, 1000, 0, 5.0).Face()
            faces.append(F)
    sew = BRepOffsetAPI_Sewing(1e-3, True, True, False, False)
    for f in faces: sew.Add(f)
    sew.Perform()
    shape = sew.SewedShape()
    return shape
