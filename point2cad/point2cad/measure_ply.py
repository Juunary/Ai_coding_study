import argparse
import open3d as o3d

def compute_surface_area_and_volume(mesh):
    # 삼각형인지 확인 필요 없이 get_surface_area/get_volume 사용
    surface_area = mesh.get_surface_area()

    try:
        volume = mesh.get_volume()
    except RuntimeError:
        print("[경고] 메쉬가 watertight하지 않아 부피 계산 불가")
        volume = None

    return surface_area, volume

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY 파일에서 겉넓이 및 부피 계산")
    parser.add_argument('--path_in', type=str, required=True, help="입력 PLY 파일 경로")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.path_in)

    if not mesh.is_watertight():
        print("[경고] 메쉬가 watertight 하지 않음 → 부피 계산이 부정확할 수 있음")

    area, vol = compute_surface_area_and_volume(mesh)

    print(f"✅ Surface Area: {area:.6f}")
    if vol is not None:
        print(f"✅ Volume: {vol:.6f}")
    else:
        print("❌ Volume: 계산 실패 (watertight 아님)")
