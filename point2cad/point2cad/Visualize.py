import numpy as np
import plotly.graph_objects as go
import trimesh

def visualize(title, path):
    # .ply 파일 로드
    mesh = trimesh.load(path)

    # Plotly로 3D 시각화
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=1.0,
                color='lightblue'
            )
        ],
        layout=dict(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            margin=dict(t=40, b=10, l=0, r=0),
        )
    )
    fig.show()

# Point2CAD 결과 파일 경로 (본인 폴더 맞춰서 수정!)
unclipped_path = "C:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/out_Data/unclipped/mesh_cube.ply"

clipped_path = "C:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/out_Data/clipped/mesh_cube.ply"

befor_path = "C:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/out_Data/clipped/mesh_mona.ply"

fix_path = "C:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/out/prior_based_mesh.ply"

visualize("Unclipped Surface", "C:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/out_Data/unclipped/mesh_mouse.ply")
visualize("Clipped Surface", "C:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/out_Data/clipped/mesh_mouse.ply")

# 시각화 실행
visualize("Unclipped Surface", unclipped_path)
visualize("Clipped Surface", clipped_path)
visualize("Before", befor_path)
visualize("After", fix_path)