"""R0.2/R2.2: BEM decimation must distinguish environment failure (missing
backend module) from genuine geometry failure, and generate_hull exports a
pre-decimated BEM STL so per-eval decimation is a fallback.

A production run of 320 designs died with generic E_RAO errors because
fast_simplification was missing from the interpreter running the pipeline:
both decimation backends raised ImportError, which was swallowed at
logger.debug and surfaced as an opaque E_RAO. These tests pin the
BemDecimationEnvError / E_RAO_ENV wiring and the generation-time export.
"""
import numpy as np
import pytest
import meshio
import tempfile
from pathlib import Path


@pytest.fixture
def config():
    from hull_opt.config import load_config
    return load_config("config.yaml")


@pytest.fixture
def dense_mesh():
    """82k-face icosphere — needs decimation for any sane BEM target."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=6)
    return meshio.Mesh(
        points=mesh.vertices.astype(np.float64),
        cells=[("triangle", mesh.faces.astype(np.int32))],
    )


def _module_not_found(*args, **kwargs):
    raise ModuleNotFoundError("No module named 'fast_simplification'")


def _generic_failure(*args, **kwargs):
    raise RuntimeError("simulated non-manifold input")


class TestBemDecimationEnvError:
    def test_missing_module_raises_env_error(self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import BemDecimationEnvError, _decimate_bem_mesh

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _module_not_found)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _module_not_found)
        with pytest.raises(BemDecimationEnvError, match="fast_simplification"):
            _decimate_bem_mesh(dense_mesh, config)

    def test_env_error_message_names_missing_module_and_fix(self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import BemDecimationEnvError, _decimate_bem_mesh

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _module_not_found)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _module_not_found)
        with pytest.raises(BemDecimationEnvError) as excinfo:
            _decimate_bem_mesh(dense_mesh, config)
        msg = str(excinfo.value)
        assert "fast_simplification" in msg
        assert "/home/anon/apps/boat/venv/bin/pip install fast_simplification" in msg

    def test_env_error_is_runtime_error_not_value_error(self):
        from hull_opt.low_fidelity import BemDecimationEnvError
        assert issubclass(BemDecimationEnvError, RuntimeError)
        assert not issubclass(BemDecimationEnvError, ValueError)

    def test_genuine_failures_still_raise_value_error(self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import BemDecimationEnvError, _decimate_bem_mesh

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _generic_failure)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _generic_failure)
        with pytest.raises(ValueError) as excinfo:
            _decimate_bem_mesh(dense_mesh, config)
        assert "could not be decimated" in str(excinfo.value)
        with pytest.raises(ValueError):
            _decimate_bem_mesh(dense_mesh, config)

    def test_mixed_failure_import_wins_to_env_error(self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import BemDecimationEnvError, _decimate_bem_mesh

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _module_not_found)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _generic_failure)
        with pytest.raises(BemDecimationEnvError):
            _decimate_bem_mesh(dense_mesh, config)

    def test_successful_backend_still_returns_mesh(self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import _decimate_bem_mesh

        def good_backend(verts, faces, n_target):
            import trimesh
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            d = m.simplify_quadric_decimation(face_count=n_target)
            return d.vertices, d.faces

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _module_not_found)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", good_backend)
        out = _decimate_bem_mesh(dense_mesh, config)
        assert isinstance(out, meshio.Mesh)
        assert len(out.cells_dict["triangle"]) <= 2500

    def test_eval_records_env_error_code(self, config, monkeypatch):
        """Environment-caused failures surface as E_RAO_ENV, not E_RAO."""
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import evaluate_low_fidelity
        import trimesh as _tm

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _module_not_found)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _module_not_found)
        big = _tm.creation.icosphere(subdivisions=5)
        monkeypatch.setattr(meshio, "read", lambda path: meshio.Mesh(
            points=big.vertices.astype(np.float64),
            cells=[("triangle", big.faces.astype(np.int32))],
        ))
        result = evaluate_low_fidelity(
            np.zeros(17), config,
            output_dir=tempfile.mkdtemp(prefix="decim_env_test_"),
            bem_skip=False)
        assert result.error_code is not None
        assert result.error_code.startswith("E_RAO_ENV:")

    def test_genuine_failure_keeps_rao_error_code(self, config, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import evaluate_low_fidelity
        import trimesh as _tm

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _generic_failure)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _generic_failure)
        big = _tm.creation.icosphere(subdivisions=5)
        monkeypatch.setattr(meshio, "read", lambda path: meshio.Mesh(
            points=big.vertices.astype(np.float64),
            cells=[("triangle", big.faces.astype(np.int32))],
        ))
        result = evaluate_low_fidelity(
            np.zeros(17), config,
            output_dir=tempfile.mkdtemp(prefix="decim_env_test_"),
            bem_skip=False)
        assert result.error_code is not None
        assert result.error_code.startswith("E_RAO:")
        assert not result.error_code.startswith("E_RAO_ENV")


class TestGenerationTimeDecimation:
    def test_generate_hull_exports_decimated_bem_stl(self):
        import trimesh
        from hull_opt.geometry import generate_hull
        from hull_opt.config import load_config

        cfg = load_config("config.yaml")
        raw = np.array([
            2.40, 0.50, 0.20, 0.60, 0.75, 10.0, 1.00, 0.20, 0.003, 0.45,
            0.20, 0.80, 12.0, 0.10, 0.005, 0.55, 0.42,
        ])
        with tempfile.TemporaryDirectory() as tmp:
            stl_path, sac_path, hydro, hull_stl = generate_hull(
                raw, output_dir=tmp, config=cfg, bem_max_faces=1200)
            bem_stl = str(Path(tmp) / "hull_geometry_bem.stl")
            assert Path(bem_stl).is_file()
            assert Path(hull_stl).is_file()
            full = trimesh.load(hull_stl)
            dec = trimesh.load(bem_stl)
            assert len(full.faces) > 1200
            assert len(dec.faces) <= 1200
            assert dec.volume > 0

    def test_generate_hull_config_target_used_by_default(self):
        import trimesh
        from hull_opt.geometry import generate_hull
        from hull_opt.config import load_config

        cfg = load_config("config.yaml")
        raw = np.array([
            2.40, 0.50, 0.20, 0.60, 0.75, 10.0, 1.00, 0.20, 0.003, 0.45,
            0.20, 0.80, 12.0, 0.10, 0.005, 0.55, 0.42,
        ])
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(raw, output_dir=tmp, config=cfg)
            bem_stl = str(Path(tmp) / "hull_geometry_bem.stl")
            assert Path(bem_stl).is_file()
            dec = trimesh.load(bem_stl)
            assert len(dec.faces) <= 2500

    def test_decimation_failure_skips_export_without_crash(self, monkeypatch):
        """Decimation failure at generation time must not break generate_hull."""
        from hull_opt.geometry import generate_hull
        from hull_opt.config import load_config

        cfg = load_config("config.yaml")
        raw = np.array([
            2.40, 0.50, 0.20, 0.60, 0.75, 10.0, 1.00, 0.20, 0.003, 0.45,
            0.20, 0.80, 12.0, 0.10, 0.005, 0.55, 0.42,
        ])
        import hull_opt.low_fidelity as lf

        monkeypatch.setattr(lf, "_decimate_fast_simplification", _generic_failure)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", _generic_failure)
        with tempfile.TemporaryDirectory() as tmp:
            stl_path, sac_path, hydro, hull_stl = generate_hull(
                raw, output_dir=tmp, config=cfg, bem_max_faces=1200)
            assert Path(hull_stl).is_file()
            assert not (Path(tmp) / "hull_geometry_bem.stl").exists()
