import numpy as np
import trimesh
import tempfile
from pathlib import Path
from hull_opt.geometry_validator import validate_hull_geometry, validate_design_vector


def _make_cube_mesh(path: str, side: float = 0.5):
    mesh = trimesh.creation.box(extents=[side, side, side])
    mesh.export(path)
    return path


def test_validates_good_mesh():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "good.stl")
        _make_cube_mesh(path, 0.5)
        is_valid, msg = validate_hull_geometry(path)
        assert is_valid, f"Expected valid, got: {msg}"


def test_rejects_null_mesh():
    is_valid, msg = validate_hull_geometry("/nonexistent/file.stl")
    assert not is_valid


def test_rejects_too_small_volume():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tiny.stl")
        _make_cube_mesh(path, 0.01)
        is_valid, msg = validate_hull_geometry(path)
        assert not is_valid
        assert "Volume too small" in msg or "Non-positive" in msg


def test_rejects_too_large_volume():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "huge.stl")
        _make_cube_mesh(path, 2.0)
        is_valid, msg = validate_hull_geometry(path)
        assert not is_valid
        assert "Volume too large" in msg


def test_rejects_too_short_hull():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "short.stl")
        mesh = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
        mesh.export(path)
        is_valid, msg = validate_hull_geometry(path)
        assert not is_valid


def test_rejects_too_narrow_beam():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "narrow.stl")
        mesh = trimesh.creation.box(extents=[1.0, 0.05, 0.5])
        mesh.export(path)
        is_valid, msg = validate_hull_geometry(path)
        assert not is_valid
        assert "narrow" in msg


def test_rejects_non_watertight():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "open.stl")
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(path)
        is_valid, msg = validate_hull_geometry(path)
        assert not is_valid


def test_validate_design_vector_finite():
    x_dict = {"BWL": 0.5, "T_canoe": 0.2, "Cp": 0.6, "D_keel": 1.0, "keel_chord": 0.2}
    is_valid, msg = validate_design_vector(x_dict)
    assert is_valid, f"Expected valid, got: {msg}"


def test_validate_design_vector_nan():
    x_dict = {"BWL": float("nan"), "T_canoe": 0.2, "Cp": 0.6, "D_keel": 1.0, "keel_chord": 0.2}
    is_valid, msg = validate_design_vector(x_dict)
    assert not is_valid
    assert "not finite" in msg


def test_validate_design_vector_out_of_bounds():
    x_dict = {"BWL": 5.0, "T_canoe": 0.2, "Cp": 0.6, "D_keel": 1.0, "keel_chord": 0.2}
    is_valid, msg = validate_design_vector(x_dict)
    assert not is_valid
    assert "outside" in msg
