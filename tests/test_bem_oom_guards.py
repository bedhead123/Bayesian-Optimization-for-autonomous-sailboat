"""Guards preventing the Capytaine dense-matrix OOM incident.

A full NURBS tessellation (~28.5k faces, ~22k immersed) fed into Capytaine
builds dense N x N complex matrices (~7.8GB each at 22k panels) and killed
the Ray worker pool with 10.8GB RSS on 2026-08-08 (BO iteration 20, the
first bem_skip=False evaluation). These tests pin the three layers of
defense:

1. _decimate_bem_mesh never returns a mesh above BEM_MAX_PANELS; it falls
   back to trimesh quadric decimation and finally raises ValueError.
2. NaN/Inf input meshes are rejected before any solve.
3. BEM entry points refuse to solve oversized meshes.
"""
import numpy as np
import pytest
import meshio


@pytest.fixture
def config():
    from hull_opt.config import load_config
    return load_config("config.yaml")


@pytest.fixture
def dense_mesh():
    """An 82k-face icosphere: ~3x the real hull tessellation, dense enough
    that a Capytaine solve would need ~100GB if the guard failed."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=6)
    return meshio.Mesh(
        points=mesh.vertices.astype(np.float64),
        cells=[("triangle", mesh.faces.astype(np.int32))],
    )


# ── Layer 1: decimation guarantees a bounded panel count ──────────────────

class TestDecimateBoundsPanelCount:
    def test_decimates_with_fast_simplification(self, config, dense_mesh):
        from hull_opt.low_fidelity import _decimate_bem_mesh, BEM_MAX_PANELS
        out = _decimate_bem_mesh(dense_mesh, config)
        n = len(out.cells_dict["triangle"])
        assert 500 <= n <= BEM_MAX_PANELS

    def test_falls_back_to_trimesh_when_fast_simplification_raises(
            self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import _decimate_bem_mesh, BEM_MAX_PANELS

        def boom(*args, **kwargs):
            raise RuntimeError("simulated non-manifold input")

        monkeypatch.setattr(lf, "_decimate_fast_simplification", boom)
        out = _decimate_bem_mesh(dense_mesh, config)
        n = len(out.cells_dict["triangle"])
        assert 500 <= n <= BEM_MAX_PANELS

    def test_raises_never_returns_full_mesh_when_all_backends_fail(
            self, config, dense_mesh, monkeypatch):
        import hull_opt.low_fidelity as lf
        from hull_opt.low_fidelity import _decimate_bem_mesh

        def boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(lf, "_decimate_fast_simplification", boom)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", boom)
        with pytest.raises(ValueError, match="could not be decimated"):
            _decimate_bem_mesh(dense_mesh, config)

    def test_rejects_nan_vertices_before_decimation(self, config, dense_mesh):
        from hull_opt.low_fidelity import _decimate_bem_mesh
        dense_mesh.points[0:50] = np.nan
        with pytest.raises(ValueError, match="NaN/Inf"):
            _decimate_bem_mesh(dense_mesh, config)

    def test_small_mesh_passes_through_untouched(self, config):
        from hull_opt.low_fidelity import _decimate_bem_mesh
        import trimesh
        small = trimesh.creation.box()
        msh = meshio.Mesh(
            points=small.vertices.astype(np.float64),
            cells=[("triangle", small.faces.astype(np.int32))],
        )
        out = _decimate_bem_mesh(msh, config)
        assert len(out.cells_dict["triangle"]) == len(msh.cells_dict["triangle"])

    def test_caps_even_when_config_requests_more(self, config, dense_mesh):
        import dataclasses
        from hull_opt.low_fidelity import _decimate_bem_mesh, BEM_MAX_PANELS
        cfg = dataclasses.replace(
            config,
            wave_spectrum=dataclasses.replace(
                config.wave_spectrum, bem_n_panels=99999))
        out = _decimate_bem_mesh(dense_mesh, cfg)
        assert len(out.cells_dict["triangle"]) <= BEM_MAX_PANELS


# ── Layer 3: BEM entry points refuse oversized meshes ─────────────────────

class TestBemEntryPointGuards:
    @pytest.fixture
    def oversized_mesh(self):
        """20k-face icosphere, returned by a monkeypatched decimator to
        simulate the guard firing before any Capytaine solve."""
        import trimesh
        big = trimesh.creation.icosphere(subdivisions=5)
        return meshio.Mesh(
            points=big.vertices.astype(np.float64),
            cells=[("triangle", big.faces.astype(np.int32))],
        )

    def test_raos_sweep_refuses_oversized_mesh(
            self, config, oversized_mesh, monkeypatch):
        from hull_opt.low_fidelity import _compute_raos_capytaine
        import meshio as _meshio
        monkeypatch.setattr(_meshio, "read", lambda path: oversized_mesh)
        monkeypatch.setattr(
            "hull_opt.low_fidelity._decimate_bem_mesh",
            lambda msh, config: oversized_mesh)
        solver_called = []

        class FakeSolver:
            def __init__(self, *a, **k):
                solver_called.append(True)

        monkeypatch.setattr("capytaine.BEMSolver", FakeSolver)
        with pytest.raises(ValueError, match="aborting solve"):
            _compute_raos_capytaine(
                "/nonexistent.stl", config, {"BWL": 0.6, "T_canoe": 0.25},
                {"nabla": 0.145, "cg_z": -0.1}, 1.0)
        assert solver_called == []

    def test_storm_sweep_refuses_oversized_mesh(
            self, config, oversized_mesh, monkeypatch):
        from hull_opt.rapid_gates import _storm_bem_sweep
        import meshio as _meshio
        monkeypatch.setattr(_meshio, "read", lambda path: oversized_mesh)
        monkeypatch.setattr(
            "hull_opt.low_fidelity._decimate_bem_mesh",
            lambda msh, config: oversized_mesh)
        with pytest.raises(ValueError, match="aborting solve"):
            _storm_bem_sweep(
                "/nonexistent.stl", config, {"BWL": 0.6, "T_canoe": 0.25},
                {"nabla": 0.145, "cg_z": -0.1}, design_vector=None,
                surrogate=None)

    def test_eval_fails_closed_without_oom(self, config, monkeypatch):
        """End-to-end: an undecimatable mesh yields an E_RAO error_code on
        the EvaluationResult — never an unhandled crash — so the GP loop
        continues and excludes the design (error_code filters it from
        get_top_n and _propose_candidate floors -inf at -100)."""
        from hull_opt.low_fidelity import evaluate_low_fidelity
        import hull_opt.low_fidelity as lf
        import meshio as _meshio
        import trimesh as _tm

        def boom(*args, **kwargs):
            raise RuntimeError("simulated total decimation failure")

        monkeypatch.setattr(lf, "_decimate_fast_simplification", boom)
        monkeypatch.setattr(lf, "_decimate_trimesh_quadric", boom)
        # R2.2: generation-time decimation now writes hull_geometry_bem.stl,
        # so force the eval-time path by feeding an oversized mesh from read.
        big = _tm.creation.icosphere(subdivisions=5)
        monkeypatch.setattr(_meshio, "read", lambda path: _meshio.Mesh(
            points=big.vertices.astype(np.float64),
            cells=[("triangle", big.faces.astype(np.int32))],
        ))
        result = evaluate_low_fidelity(
            np.zeros(17), config, output_dir="/tmp/opencode/bem_guard_test",
            bem_skip=False)
        assert result.error_code is not None
        assert result.error_code.startswith("E_RAO")
