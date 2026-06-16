from datetime import datetime
import subprocess

now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

subprocess.run(
    [
        "python",
        "/mnt/nas4/junewookang/point2cad/assets/Test.py",
        "--path1",
        "/mnt/nas4/junewookang/point2cad/assets/casing/ply_validation",
        "--path2",
        "/mnt/nas4/junewookang/point2cad/assets/casing/ply_test",
        "--start",
        "1",
        "--end",
        "120",
        "--pattern",
        "casing_{:03d}.ply",
        "--scale",
        "1",
        "--log",
        f"/mnt/nas4/junewookang/point2cad/assets/results/evaluation_casing_{now}.log",
        "--len_eps_mm",
        "1e-300",
        "--abs_eps_mm",
        "1e-300",
    ]
)
