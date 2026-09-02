import os
import subprocess
import numpy as np
import pytest
from pathlib import Path
from dataclasses import dataclass


@pytest.fixture
def config():
    from hull_opt.config import load_config
    return load_config("config.yaml")


@pytest.fixture
def settings():
    from hull_opt.sph_gates import ReferenceSettings
    return ReferenceSettings()


@pytest.fixture
def design_vector():
    return np.array([2.4, 0.50, 0.25, 0.60, 0.75, 45.0, 1.0, 0.20, 0.0012,
                     0.40, 0.20, 12.0, 15.0, 0.15, 0.01, 0.50, 0.42],
                    dtype=np.float64)


class TestStormXmlRenders:
    def test_contains_irregular(self):
        from hull_opt.templates.dualsphysics import STORM_XML, DEF_MARGIN
        output = STORM_XML.render(
            gravity=9.81, rho=1025.0, nu=1e-6,
            sim_time=10.0, dt_out=0.1,
            dp=0.02,
            xmax=3.6, ymax=0.75, zmin=0.75, zmax=1.125,
            def_margin=DEF_MARGIN,
            xdomain=3.6, ydomain=0.75, zdomain=1.125,
            hull_stl_path="/tmp/hull.stl",
            hull_x=-1.2, hull_z=0.0,
            seed_x=0.1, seed_y=-0.5, seed_z=-0.5,
            mass=300.0, Ixx=10.0, Iyy=500.0, Izz=490.0,
            cg_str='<cg x="0" y="0" z="-0.1"/>',
            tank_depth=1.125, still_water_level=0.35,
            Hs=0.25, Tp=1.3, gamma=3.3,
            x_paddle_lo=-3.25, x_paddle_th=0.1,
            x_damp=2.52, ramp_time=0.65,
        )
        assert "piston_spectrum" in output
        assert "jonswap" in output
        assert "waveheight" in output
        assert "floating" in output
        assert "casedef" in output
        assert "massbody" in output
        assert "peakcoef" in output or "gamma" in output
        assert "pistonwave" in output or "piston" in output


@pytest.fixture
def box_stl(tmp_path):
    import trimesh
    hull = trimesh.creation.box(extents=[1.4, 0.4, 0.2], transform=[
        [1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, -0.1], [0, 0, 0, 1]])
    keel = trimesh.creation.box(extents=[0.8, 0.05, 0.5], transform=[
        [1, 0, 0, 0.8], [0, 1, 0, 0], [0, 0, 1, -0.35], [0, 0, 0, 1]])
    stl = tmp_path / "hull_test.stl"
    trimesh.util.concatenate([hull, keel]).export(str(stl))
    return str(stl)


class TestTowingXmlWrites:
    def test_writes_case_and_points(self, tmp_path, box_stl):
        from hull_opt.templates.dualsphysics import write_towing_case
        xml = write_towing_case(
            str(tmp_path / "tow"), box_stl, 1.8, 0.45, 0.18, 0.55,
            speed_ms=1.2, dp=0.01, mass=120.0,
        )
        assert xml.exists()
        text = xml.read_text()
        # v3 moving-hull tow: closed tank, hull as moving mkbound, NO InOut
        # zones (Bug #157 — imposed outlet trapped waves / flux pile-up)
        assert "inoutzone" not in text
        assert "mvrect" in text
        assert "damping" in text.lower()
        assert "drawfilestl" in text
        assert "speed" in text.lower()
        pts = (tmp_path / "tow" / "measure_points.txt").read_text()
        assert pts.startswith("POINTS")
        assert len(pts.splitlines()) == 2

    def test_no_undefined_vars(self, tmp_path, box_stl):
        import re
        from hull_opt.templates.dualsphysics import write_towing_case
        xml = write_towing_case(
            str(tmp_path / "tow2"), box_stl, 1.8, 0.45, 0.18, 0.55,
            speed_ms=1.2, dp=0.01, mass=120.0,
        )
        assert re.findall(r"\{\{\s*\w+\s*\}\}", xml.read_text()) == []


class TestInvertedXmlWrites:
    def test_writes_case_and_points(self, tmp_path, box_stl):
        from hull_opt.templates.dualsphysics import write_inverted_case
        xml = write_inverted_case(
            str(tmp_path / "inv"), box_stl, 1.8, 0.45, 0.18, 0.55,
            mass=120.0, dp=0.01,
        )
        assert xml.exists()
        text = xml.read_text()
        assert "floating" in text
        assert "massbody" in text
        assert "drawfilestl" in text
        assert (tmp_path / "inv" / "inverted_hull.stl").exists()
        pts = (tmp_path / "inv" / "measure_points.txt").read_text()
        assert pts.startswith("POINTS")
        lines = pts.splitlines()[1:]
        assert 3 <= len(lines) <= 15

    def test_measure_points_below_surface(self, tmp_path, box_stl):
        from hull_opt.templates.dualsphysics import write_inverted_case
        write_inverted_case(
            str(tmp_path / "inv2"), box_stl, 1.8, 0.45, 0.18, 0.55,
            mass=120.0, dp=0.01,
        )
        pts = (tmp_path / "inv2" / "measure_points.txt").read_text().splitlines()[1:]
        zs = [float(line.split()[2]) for line in pts if line]
        assert all(z < 0.0 for z in zs)

    def test_no_undefined_vars(self, tmp_path, box_stl):
        import re
        from hull_opt.templates.dualsphysics import write_inverted_case
        xml = write_inverted_case(
            str(tmp_path / "inv3"), box_stl, 1.8, 0.45, 0.18, 0.55,
            mass=120.0, dp=0.01,
        )
        assert re.findall(r"\{\{\s*\w+\s*\}\}", xml.read_text()) == []


def _parse_case_xml(xml_text):
    """Return (root, seed, fillbox point, fillbox size, def min, def max) from a
    rendered DualSPHysics case XML, all as {'x','y','z'} float dicts."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    fillbox = next(el for el in root.iter() if el.tag == "fillbox")
    definition = next(el for el in root.iter() if el.tag == "definition")
    p = fillbox.find("point")
    s = fillbox.find("size")
    seed = {a: float(fillbox.get(a)) for a in ("x", "y", "z")}
    point = {a: float(p.get(a)) for a in ("x", "y", "z")}
    size = {a: float(s.get(a)) for a in ("x", "y", "z")}
    def_min = {a: float(definition.find("pointmin").get(a)) for a in ("x", "y", "z")}
    def_max = {a: float(definition.find("pointmax").get(a)) for a in ("x", "y", "z")}
    return root, seed, point, size, def_min, def_max


def _render_dualwriter_case(writer_name, case_dir, hull_stl, design, extra):
    """Render one of the 5 writers with the given (LWL, B, T_canoe, D_keel)
    design and return the case XML text. storm/towing/inverted return the xml
    path; focused_wave/drop_impact return None, so glob the case file instead."""
    from hull_opt.templates import dualsphysics as dsp
    writer = getattr(dsp, writer_name)
    LWL, B, T_canoe, D_keel = design
    if writer_name in ("write_towing_case", "write_inverted_case"):
        xml_path = writer(case_dir, hull_stl, LWL, B, T_canoe, D_keel,
                          mass=150.0, dp=0.02, **extra)
    else:
        xml_path = writer(case_dir, hull_stl, LWL, B, T_canoe,
                          mass=150.0, dp=0.02, **extra)
    if xml_path is None:
        xml_path = next(Path(case_dir).glob("case_*.xml"))
    return Path(xml_path).read_text()


class TestFillboxSeedInsideBox:
    """Regression guard for the GenCase fillbox seed fix.

    The seed was previously hardcoded x="1.0" y="1.0" z="1.0" — outside the
    fill boxes — so GenCase created zero fluid particles and the solver
    crashed with "Constant 'b' cannot be zero". Seeds are now computed
    per-writer; these tests lock in that the seed is strictly inside the
    fillbox, that the fillbox itself lies within the case definition
    (pointmin..pointmax) box, and that the legacy literal never returns.
    Run at the config-bound extremes so the invariant holds across the
    design space.
    """

    _WRITERS = [
        ("storm", "write_storm_case", {}),
        ("focused_wave", "write_focused_wave_case", {}),
        ("drop_impact", "write_drop_impact_case", {}),
        ("towing", "write_towing_case", {"speed_ms": 1.65}),
        ("inverted", "write_inverted_case", {}),
    ]

    @pytest.fixture
    def hull_stl(self):
        import glob
        matches = sorted(glob.glob("/home/anon/apps/boat/output/design_*/hull.stl"))
        if not matches:
            pytest.skip("no real hull STL found at output/design_*/hull.stl")
        return matches[0]

    @pytest.fixture
    def design_extremes(self, config):
        b = config.bounds
        return {
            "min": (b.LWL[0], b.BWL[0], b.T_canoe[0], b.D_keel[0]),
            "max": (b.LWL[1], b.BWL[1], b.T_canoe[1], b.D_keel[1]),
        }

    @pytest.mark.parametrize("design_key", ["min", "max"])
    def test_seed_inside_fillbox_and_fillbox_inside_definition_box(
            self, tmp_path, hull_stl, design_extremes, design_key):
        design = design_extremes[design_key]
        for name, writer, extra in self._WRITERS:
            xml_text = _render_dualwriter_case(
                writer, tmp_path / f"{name}_{design_key}", hull_stl, design, extra)
            _, seed, point, size, def_min, def_max = _parse_case_xml(xml_text)
            for axis in ("x", "y", "z"):
                box_lo = point[axis]
                box_hi = point[axis] + size[axis]
                assert box_lo + 1e-9 < seed[axis] < box_hi - 1e-9, (
                    f"{name} ({design_key}): seed {axis}={seed[axis]} outside "
                    f"fillbox [{box_lo}, {box_hi}]")
                assert def_min[axis] - 1e-9 <= box_lo and box_hi <= def_max[axis] + 1e-9, (
                    f"{name} ({design_key}): fillbox {axis} range "
                    f"[{box_lo}, {box_hi}] outside definition [{def_min[axis]}, "
                    f"{def_max[axis]}]")
            assert 'x="1.0" y="1.0" z="1.0"' not in xml_text, (
                f"{name} ({design_key}): legacy hardcoded seed literal found")

    @pytest.mark.parametrize("design_key", ["min", "max"])
    def test_towing_fillbox_depth_is_shrunken_tank(
            self, tmp_path, hull_stl, design_extremes, design_key):
        design = design_extremes[design_key]
        xml_text = _render_dualwriter_case(
            "write_towing_case", tmp_path / f"tow_depth_{design_key}",
            hull_stl, design, {"speed_ms": 1.65})
        root, seed, point, size, _, _ = _parse_case_xml(xml_text)
        zsurf = float(next(el for el in root.iter()
                           if el.tag == "zsurf").get("value"))
        assert size["z"] == pytest.approx(zsurf - point["z"], abs=1e-9)
        T_total = design[2] + design[3]
        assert size["z"] == pytest.approx(1.5 * T_total, abs=1e-9), (
            f"towing tank depth {size['z']} != 1.5*T_total = {1.5 * T_total}")

    def test_no_legacy_hardcoded_seed_in_templates(self):
        import inspect
        from hull_opt.templates import dualsphysics as dsp
        assert len(dsp.TEMPLATES) == 5
        source = inspect.getsource(dsp)
        fillbox_lines = [line for line in source.splitlines() if "<fillbox" in line]
        assert len(fillbox_lines) == 5
        for line in fillbox_lines:
            assert "{{ seed_" in line, f"fillbox uses no seed var: {line.strip()}"
            assert 'x="1.0" y="1.0" z="1.0"' not in line, (
                f"legacy hardcoded seed still in template: {line.strip()}")


class TestHullPlacementKeelAware:
    """Verifies that the hull is placed at the correct draft in all five
    SPH cases, and that the full-mesh STL (hull+keel+bulb) is preferred
    when available.

    The fixtures use real hull STLs from output/design_*/hull.stl
    (hull-only).  Since bug #141 the writers resolve the STL through
    _sph_stl_path() (preferring the sibling hull_full.stl with keel/bulb)
    and place the DEEPEST POINT OF THAT MESH at the design draft:
    hull_z = (still_water_level - T) - stl_z_min for storm/focused
    (still_water_level = T + 0.1, so dz + z_min = 0.1), and
    init_z = drop_height - stl_z_min for the drop case.  The placement
    invariant must be asserted against the bounds of the mesh the writer
    actually used, not the hull-only fixture.
    """

    @pytest.fixture
    def hull_stl_path(self):
        import glob
        matches = sorted(glob.glob("/home/anon/apps/boat/output/design_*/hull.stl"))
        if not matches:
            pytest.skip("no real hull STL at output/design_*/hull.stl")
        return matches[0]

    def test_all_writers_produce_drawmove(self, hull_stl_path):
        """All five SPH writers must produce an XML with a drawmove element."""
        import xml.etree.ElementTree as ET
        design = (2.4, 0.5, 0.25, 1.0)
        for name, writer, extra in [
            ("storm", "write_storm_case", dict()),
            ("focused_wave", "write_focused_wave_case",
             dict(wave_height=0.25, wave_period=1.3)),
            ("drop", "write_drop_impact_case", dict(drop_height=2.0)),
            ("towing", "write_towing_case", dict(speed_ms=1.65)),
            ("inverted", "write_inverted_case", dict()),
        ]:
            txt = _render_dualwriter_case(
                writer, Path(f"/tmp/sph_test_dm_{name}_{id(self)}"),
                hull_stl_path, design, extra)
            root = ET.fromstring(txt)
            draw = root.find(".//drawmove")
            assert draw is not None, f"{name}: no drawmove element"
            assert "x" in draw.attrib and "z" in draw.attrib

    def test_storm_focused_keel_tip_at_draft(self, hull_stl_path):
        """Storm/focused-wave drawmove z places the deepest point of the mesh
        the writer actually uses (hull_full.stl via _sph_stl_path) at
        still_water_level - T = 0.1 m.  Regression: the test previously read
        the hull-only fixture STL bounds while the writer places the full
        keel/bulb mesh, so the assertion was stale."""
        from hull_opt.templates.dualsphysics import _sph_stl_path
        import trimesh
        used_stl = _sph_stl_path(hull_stl_path)
        m = trimesh.load(used_stl, force="mesh")
        z_min = float(m.bounds[0, 2])
        import xml.etree.ElementTree as ET
        design = (2.4, 0.5, 0.25, 1.0)
        for name, writer, extra in [
            ("storm", "write_storm_case", dict()),
            ("focused_wave", "write_focused_wave_case",
             dict(wave_height=0.25, wave_period=1.3)),
        ]:
            txt = _render_dualwriter_case(
                writer, Path(f"/tmp/sph_test_draft_{name}_{id(self)}"),
                hull_stl_path, design, extra)
            root = ET.fromstring(txt)
            dz = float(root.find(".//drawmove").attrib["z"])
            # deepest point of the placed mesh at still_water_level - T = 0.1
            assert dz + z_min == pytest.approx(0.1, abs=2e-3), (
                f"{name}: hull_z={dz:.4f} + stl_zmin={z_min:.4f} = {dz+z_min:.4f} ≠ 0.1 "
                f"(placed mesh: {Path(used_stl).name})")

    def test_drop_keel_tip_at_drop_height(self, hull_stl_path):
        """Drop writer init_z places the deepest point of the placed mesh
        (hull_full.stl via _sph_stl_path) at the given drop_height above the
        water surface (zsurf=0)."""
        from hull_opt.templates.dualsphysics import _sph_stl_path
        import trimesh
        used_stl = _sph_stl_path(hull_stl_path)
        m = trimesh.load(used_stl, force="mesh")
        z_min = float(m.bounds[0, 2])
        drop_h = 2.0
        import xml.etree.ElementTree as ET
        txt = _render_dualwriter_case(
            "write_drop_impact_case",
            Path(f"/tmp/sph_test_drop_{id(self)}"),
            hull_stl_path, (2.4, 0.5, 0.25, 1.0),
            dict(drop_height=drop_h))
        root = ET.fromstring(txt)
        init_z = float(root.find(".//drawmove").attrib["z"])
        assert init_z + z_min == pytest.approx(drop_h, abs=2e-3), (
            f"drop: init_z={init_z:.4f} + stl_zmin={z_min:.4f} = {init_z+z_min:.4f} ≠ {drop_h} "
            f"(placed mesh: {Path(used_stl).name})")

    def test_sph_stl_path_prefers_full_mesh(self, tmp_path):
        """_sph_stl_path returns the sibling hull_full.stl when present,
        and falls back to the given path otherwise."""
        from hull_opt.templates.dualsphysics import _sph_stl_path
        hull = tmp_path / "hull.stl"
        hull.write_text("dummy")
        full = tmp_path / "hull_full.stl"
        assert _sph_stl_path(str(hull)) == str(hull.resolve())
        full.write_text("dummy")
        assert _sph_stl_path(str(hull)) == str(full.resolve())

    def test_storm_and_focused_templates_have_drawmove_block(self):
        """The STORM_XML and FOCUSED_WAVE_XML Jinja templates contain a
        drawmove block (regression guard against accidental removal)."""
        from hull_opt.templates.dualsphysics import STORM_XML, FOCUSED_WAVE_XML, DEF_MARGIN
        for name, tmpl in [("storm", STORM_XML), ("focused_wave", FOCUSED_WAVE_XML)]:
            source = tmpl.render(
                hull_stl_path="x", hull_x=0, hull_z=0,
                xmax=1, ymax=1, zmin=1, zmax=1,
                def_margin=DEF_MARGIN,
                xdomain=1, ydomain=1, zdomain=1,
                seed_x=0, seed_y=0, seed_z=0,
                mass=150, Ixx=1, Iyy=1, Izz=1, cg_str="",
                tank_depth=1, still_water_level=0,
                Hs=0.25, Tp=1.3, gamma=3.3,
                x_paddle_lo=-0.65, x_paddle_th=0.1,
                x_damp=0.7, ramp_time=0.65,
            )
            assert "<drawmove" in source, f"{name} template missing drawmove"


class TestFindSolver:
    def test_gpu(self, monkeypatch):
        def mock_exists(path):
            return "DSGcc7/DualSPHysics54.linux64" in str(path)
        monkeypatch.setattr(os.path, "exists", mock_exists)
        from hull_opt.sph_gates import find_solver
        path, kind, env = find_solver("/opt/dualsphysics/5.4")
        assert kind == "gpu"
        assert "DualSPHysics54.linux64" in path
        assert isinstance(env, dict)
        assert env["LD_LIBRARY_PATH"] == str(Path(path).parent)

    def test_fallback_cpu(self, monkeypatch):
        def mock_exists(path):
            if "DSGcc7/DualSPHysics54.linux64" in str(path):
                return False
            return "DualSPHysics5.4CPU_linux64" in str(path)
        monkeypatch.setattr(os.path, "exists", mock_exists)
        from hull_opt.sph_gates import find_solver
        path, kind, env = find_solver("/opt/dualsphysics/5.4")
        assert kind == "cpu"
        assert "CPU" in path
        assert isinstance(env, dict)

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        from hull_opt.sph_gates import find_solver
        with pytest.raises(FileNotFoundError):
            find_solver("/opt/dualsphysics/5.4")


class TestRun:
    def test_timeout(self, config, settings, design_vector, monkeypatch):
        import tempfile
        case_dir = tempfile.mkdtemp()

        class FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        def mock_run(*args, **kwargs):
            # First call is GenCase: create gencase.xml so the existence check
            # passes, then the solver call will time out.
            gc_xml = Path(case_dir) / "gencase.xml"
            if not gc_xml.exists():
                gc_xml.write_text("<case/>")
                return FakeProc()
            raise subprocess.TimeoutExpired(cmd="test", timeout=1)

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        from hull_opt.sph_gates import run_reference_storm
        result = run_reference_storm(design_vector, config, case_dir,
                                     "test_to", settings=settings,
                                     stl_path="/tmp/hull.stl")
        assert result["status"] == "TIMEOUT"
        assert result["wall_time_s"] >= 0

    def test_skipped(self, config, settings, design_vector, monkeypatch):
        import fcntl
        orig_flock = fcntl.flock
        def mock_flock(fd, op):
            if op & fcntl.LOCK_NB:
                raise BlockingIOError(11, "Resource temporarily unavailable")
            return orig_flock(fd, op)
        monkeypatch.setattr(fcntl, "flock", mock_flock)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        from hull_opt.sph_gates import run_reference_storm
        result = run_reference_storm(design_vector, config, "/tmp/sph_test_sk",
                                     "test_sk", settings=settings, stl_path="/tmp/hull.stl",
                                     gpu_lock="/tmp/test_gpu_lock")
        assert result["status"] == "SKIPPED"


class TestMeasureToolCsvParse:
    @pytest.fixture
    def csv_content(self):
        return (
            "Time;AccelX;AccelY;AccelZ;Pressure\n"
            "0.0;0.0;0.0;0.0;0.0\n"
            "0.1;5.0;2.0;9.81;1000.0\n"
            "0.2;15.0;3.0;29.43;5000.0\n"
            "0.3;10.0;1.0;19.62;3000.0\n"
        )

    def test_parse_known_values(self, tmp_path, csv_content):
        from hull_opt.sph_gates import _parse_measuretool_csv
        csv_file = tmp_path / "MeasureTool.csv"
        csv_file.write_text(csv_content)
        result = _parse_measuretool_csv(csv_file)
        assert result["max_accel_g"] == pytest.approx(29.43 / 9.81, rel=1e-3)
        assert result["max_pressure_pa"] == 5000.0

    def test_missing_file(self, tmp_path):
        from hull_opt.sph_gates import _parse_measuretool_csv
        result = _parse_measuretool_csv(tmp_path / "nonexistent.csv")
        assert np.isnan(result["max_accel_g"])
        assert np.isnan(result["max_pressure_pa"])

    def test_empty_csv(self, tmp_path):
        from hull_opt.sph_gates import _parse_measuretool_csv
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("# Time;Accel\n")
        result = _parse_measuretool_csv(csv_file)
        assert np.isnan(result["max_accel_g"]) or result["max_accel_g"] == 0.0
        assert np.isnan(result["max_pressure_pa"]) or result["max_pressure_pa"] == 0.0


class TestRunOk:
    def test_ok_metrics(self, config, settings, design_vector, monkeypatch, tmp_path):
        def mock_find_solver(*args, **kwargs):
            return ("/fake/solver", "cpu", os.environ.copy())
        monkeypatch.setattr("hull_opt.sph_gates.find_solver", mock_find_solver)

        def mock_find_tool(*args, **kwargs):
            return "/fake/tool"
        monkeypatch.setattr("hull_opt.sph_gates._find_tool", mock_find_tool)

        import subprocess as sp
        class FakeProc:
            returncode = 0
            stdout = ""
            stderr = ""

        case_dir = tmp_path / "storm_ok"

        def mock_run(*args, **kwargs):
            if case_dir.exists():
                (case_dir / "gencase.xml").write_text("<case/>")
            return FakeProc()

        monkeypatch.setattr(sp, "run", mock_run)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        from hull_opt.sph_gates import run_reference_storm
        result = run_reference_storm(design_vector, config, str(case_dir),
                                     "test_ok", settings=settings, stl_path="/tmp/hull.stl")
        assert result["status"] == "OK"
        assert result["tool"] == "dualsphysics"
        assert "max_accel_g" in result
        assert "max_pressure_pa" in result
        assert "capsized" in result
        assert "wall_time_s" in result
        assert "sim_time_s" in result
        assert "details" in result
        assert isinstance(result["capsized"], bool)
        assert result["wall_time_s"] >= 0


class TestRunStormGuards:
    """Particle-cap + VRAM guard + GPU flag on run_reference_storm (mirrors
    run_sph_case): oversized storms die in seconds, never on the GPU."""

    def _mock_env(self, monkeypatch, tmp_path, gencase_line, free_mib=None):
        case_dir = tmp_path / "storm_guard"
        case_dir.mkdir(parents=True, exist_ok=True)
        calls = []

        def mock_find_solver(*args, **kwargs):
            return ("/fake/solver", "gpu", os.environ.copy())

        def mock_find_tool(*args, **kwargs):
            return "/fake/tool"

        def mock_run(cmd, **kw):
            calls.append(list(cmd))
            if len(calls) == 1:
                (case_dir / "gencase.xml").write_text("<case/>")
                (case_dir / "gencase.out").write_text(gencase_line)
            if cmd[0] == "nvidia-smi":
                return type("P", (), {"returncode": 0,
                                      "stdout": f"{free_mib}\n", "stderr": ""})()
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("hull_opt.sph_gates.find_solver", mock_find_solver)
        monkeypatch.setattr("hull_opt.sph_gates._find_tool", mock_find_tool)
        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        return case_dir, calls

    def test_oversized_case_fails_before_solver(self, config, settings,
                                                design_vector, monkeypatch,
                                                tmp_path):
        case_dir, calls = self._mock_env(
            monkeypatch, tmp_path, "Total particles: 9,018,952 (bound=1 fluid=9018951)")
        from hull_opt.sph_gates import run_reference_storm
        result = run_reference_storm(design_vector, config, str(case_dir),
                                     "test_cap", settings=settings,
                                     stl_path="/tmp/hull.stl")
        assert result["status"] == "FAILED"
        assert "exceeds cap" in result["details"]
        assert result["wall_time_s"] < 60, "must die in seconds, not hours"
        solver_calls = [c for c in calls if "-dirout" in c]
        assert not solver_calls, "solver must never be launched for oversized case"

    def test_under_cap_launches_gpu_solver(self, config, settings,
                                           design_vector, monkeypatch,
                                           tmp_path):
        case_dir, calls = self._mock_env(
            monkeypatch, tmp_path, "Total particles: 284,235 (bound=27170 fluid=257065)",
            free_mib=6000)
        from hull_opt.sph_gates import run_reference_storm
        result = run_reference_storm(design_vector, config, str(case_dir),
                                     "test_gpu", settings=settings,
                                     stl_path="/tmp/hull.stl")
        assert result["status"] == "OK"
        solver_calls = [c for c in calls if "-dirout" in c]
        assert len(solver_calls) == 1
        assert "-gpu" in solver_calls[0], (
            "GPU solver binary must be launched with -gpu (default is CPU "
            "per JSphCfgRun.cpp; without it storms crawl)")

    def test_vram_guard_aborts(self, config, settings, design_vector,
                               monkeypatch, tmp_path):
        case_dir, calls = self._mock_env(
            monkeypatch, tmp_path, "Total particles: 284,235 (bound=27170 fluid=257065)",
            free_mib=100)
        from hull_opt.sph_gates import run_reference_storm
        result = run_reference_storm(design_vector, config, str(case_dir),
                                     "test_vram", settings=settings,
                                     stl_path="/tmp/hull.stl")
        assert result["status"] == "FAILED"
        assert "VRAM guard" in result["details"]
        solver_calls = [c for c in calls if "-dirout" in c]
        assert not solver_calls


class TestAdaptiveStormDp:
    """Design-adaptive reference-storm dp: oversized hulls (designs 11/41 in
    the production run: 501,868 / 505,841 particles at dp=0.06) must be
    coarsened just enough to stay under the 500k cap; small hulls keep the
    nominal dp; bad inputs fall back without crashing."""

    def test_oversized_hull_coarsened_below_cap(self):
        from hull_opt.sph_gates import (_STORM_MAX_PARTICLES,
                                        _estimate_storm_dp,
                                        _storm_defbox_volume)
        # Design 11: LWL=2.4289, BWL=0.7000, T_total=1.5228 (real: 501,868)
        dp = _estimate_storm_dp(2.4289, 0.7000, 1.5228, 0.25, 0.06)
        assert dp > 0.06, "oversized hull must be coarsened"
        assert dp <= 0.10
        assert dp == 0.064, "ceil to 3 decimals"
        est = 0.65 * _storm_defbox_volume(2.4289, 0.7000, 1.5228, 0.25, dp) \
            / dp ** 3
        assert est <= _STORM_MAX_PARTICLES, "estimate must clear the cap"

    def test_second_oversized_hull(self):
        from hull_opt.sph_gates import (_STORM_MAX_PARTICLES,
                                        _estimate_storm_dp,
                                        _storm_defbox_volume)
        # Design 41: LWL=2.4741, BWL=0.6740, T_total=1.5432 (real: 505,841)
        dp = _estimate_storm_dp(2.4741, 0.6740, 1.5432, 0.25, 0.06)
        assert dp > 0.06
        assert dp == 0.064
        est = 0.65 * _storm_defbox_volume(2.4741, 0.6740, 1.5432, 0.25, dp) \
            / dp ** 3
        assert est <= _STORM_MAX_PARTICLES

    def test_small_hull_keeps_nominal_dp(self):
        from hull_opt.sph_gates import _estimate_storm_dp
        # Design 5 (real dims: LWL=2.4996, BWL=0.5026, T_total=1.5499; real
        # GenCase layout 389,673 < cap): must stay at dp=0.06 untouched.
        assert _estimate_storm_dp(2.4996, 0.5026, 1.5499, 0.25, 0.06) == 0.06

    def test_non_finite_falls_back_to_nominal(self):
        from hull_opt.sph_gates import _estimate_storm_dp
        assert _estimate_storm_dp(float("nan"), 0.5, 1.1, 0.25, 0.06) == 0.06
        assert _estimate_storm_dp(2.4, float("inf"), 1.1, 0.25, 0.06) == 0.06
        assert _estimate_storm_dp(2.4, 0.5, 1.1, 0.25, -0.1) == -0.1

    def test_extreme_hull_capped_at_max_dp(self):
        from hull_opt.sph_gates import (_STORM_MAX_DP,
                                        _STORM_MAX_PARTICLES,
                                        _estimate_storm_dp,
                                        _storm_defbox_volume)
        dp = _estimate_storm_dp(2.5, 0.7, 3.0, 0.25, 0.06)
        assert 0.06 < dp <= _STORM_MAX_DP
        est = 0.65 * _storm_defbox_volume(2.5, 0.7, 3.0, 0.25, dp) / dp ** 3
        assert est <= _STORM_MAX_PARTICLES
        assert est <= 480_000, "targeted near 420k, not just under the cap"
