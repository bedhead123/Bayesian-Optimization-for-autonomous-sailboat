import numpy as np
import trimesh
import tempfile
from hull_opt.geometry import generate_hull
from hull_opt.config import load_config

config = load_config("config.yaml")

DESIGN = np.array([
    2.40, 0.50, 0.20, 0.60, 0.75, 10.0, 1.00, 0.20,
    0.003, 0.45, 0.20, 0.80, 12.0, 0.10, 0.005,
    0.55, 0.42,
])

with tempfile.TemporaryDirectory() as tmp:
    stl_path, sac_path, hydro, hull_stl = generate_hull(DESIGN, output_dir=tmp, config=config)
    mesh = trimesh.load(hull_stl)

    volume = mesh.volume
    n_faces = len(mesh.faces)

    centroids = mesh.triangles_center
    normals = mesh.face_normals

    port_mask = centroids[:, 1] < 0
    port_normals_y = normals[port_mask, 1]

    n_port = port_mask.sum()
    n_port_pos_y = (port_normals_y > 0).sum()
    n_port_neg_y = (port_normals_y < 0).sum()

    print(f"Volume: {volume:.6f}  (sign: {'+' if volume > 0 else '-'})")
    print(f"Total faces: {n_faces}")
    print(f"Port-side faces: {n_port}")
    print(f"  Port normals with +y (inward on port): {n_port_pos_y}")
    print(f"  Port normals with -y (outward on port): {n_port_neg_y}")
    print(f"  Dominant direction: {'+y (INWARD)' if n_port_pos_y > n_port_neg_y else '-y (OUTWARD)'}")
