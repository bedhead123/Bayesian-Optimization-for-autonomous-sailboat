"""Preflight behavior: WARN (not FAIL) on non-watertight STL, FAIL on
missing/unloadable/degenerate STL. DualSPHysics does not require watertight
meshes (hull_full.stl has open keel/bulb patches by design)."""
from pathlib import Path

import trimesh

from hull_opt.config import load_config
from hull_opt.preflight import preflight_case

_CONFIG = load_config("config.yaml")


def _render_case(tmp_path, stl_path: Path, name: str = "case_test.xml") -> Path:
    case_dir = tmp_path / "case"
    case_dir.mkdir(exist_ok=True)
    xml = case_dir / name
    xml.write_text(f'<case><geometry><triangulation file="{stl_path.resolve()}"/></geometry></case>')
    return case_dir


def _open_box(path: Path) -> Path:
    box = trimesh.creation.box(extents=(1.0, 0.5, 0.3))
    box.faces = box.faces[:-2]  # drop the last face -> open boundary
    box.remove_unreferenced_vertices()
    assert not box.is_watertight, "test mesh must be non-watertight"
    box.export(path)
    return path


def test_non_watertight_stl_warns_but_passes(tmp_path):
    stl = _open_box(tmp_path / "open_box.stl")
    case_dir = _render_case(tmp_path, stl)
    pf_ok, pf_msgs = preflight_case(case_dir, _CONFIG)
    assert pf_ok, "\n".join(pf_msgs)
    watertight_msgs = [m for m in pf_msgs if "watertight" in m.lower()]
    assert watertight_msgs and all(m.startswith("WARN") for m in watertight_msgs), pf_msgs


def test_watertight_stl_passes(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 0.5, 0.3))
    assert box.is_watertight
    stl = tmp_path / "closed_box.stl"
    box.export(stl)
    case_dir = _render_case(tmp_path, stl)
    pf_ok, pf_msgs = preflight_case(case_dir, _CONFIG)
    assert pf_ok, "\n".join(pf_msgs)
    assert any("watertight" in m and m.startswith("PASS") for m in pf_msgs), pf_msgs


def test_missing_stl_fails(tmp_path):
    case_dir = _render_case(tmp_path, tmp_path / "missing.stl")
    pf_ok, pf_msgs = preflight_case(case_dir, _CONFIG)
    assert not pf_ok
    assert any(m.startswith("FAIL") and "STL path exists" in m for m in pf_msgs), pf_msgs


def test_degenerate_stl_fails(tmp_path):
    mesh = trimesh.Trimesh(
        vertices=[[1.0, 2.0, 3.0]] * 3,
        faces=[[0, 1, 2]],
        process=False,
    )
    stl = tmp_path / "degenerate.stl"
    mesh.export(stl)
    case_dir = _render_case(tmp_path, stl)
    pf_ok, pf_msgs = preflight_case(case_dir, _CONFIG)
    assert not pf_ok
    assert any(m.startswith("FAIL") and "degenerate" in m for m in pf_msgs), pf_msgs


def test_no_case_xml_fails(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    pf_ok, pf_msgs = preflight_case(empty, _CONFIG)
    assert not pf_ok
    assert any(m.startswith("FAIL") and "Case XML file found" in m for m in pf_msgs), pf_msgs
