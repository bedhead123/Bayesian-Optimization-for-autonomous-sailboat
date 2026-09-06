"""Write output/results_clean.csv — human-readable leaderboard.

One row per feasible design, sorted by mission_fom desc. Rounded numbers,
key specs only, validation status, winner flagged. Usage:
    venv/bin/python scripts/write_clean_csv.py
"""
import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, ".")

COLUMNS = [
    "id", "iter", "mission_fom", "fom",
    "LWL_m", "BWL_m", "BWL_measured_m", "T_canoe_m", "D_keel_m", "keel_chord_m",
    "bulb_L", "ballast_frac", "T_over_L", "LDR",
    "GM_m", "AVS_deg", "righting_energy_J",
    "heavy_leeway_deg", "helm_comb_deg", "mission_drive",
    "gust_margin", "draft_logistics", "Rt_N",
    "validation", "best",
]


def r3(v):
    try:
        return round(float(v), 3) if v is not None else ""
    except Exception:
        return ""


def r1(v):
    try:
        return round(float(v), 1) if v is not None else ""
    except Exception:
        return ""


def main():
    db_path = Path("output/optimization.db")
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cols = [d[1] for d in c.execute("pragma table_info(designs)")]
    has_mission = "mission_fom" in cols
    order = "mission_fom DESC" if has_mission else "fom DESC"
    rows = c.execute(
        f"SELECT * FROM designs WHERE feasible = 1 ORDER BY {order}"
    ).fetchall()
    drows = [dict(zip(cols, r)) for r in rows]
    val = {}
    try:
        for did, g, p in c.execute(
                "SELECT design_id, gate_name, passed FROM validation"):
            val.setdefault(did, []).append((g, p))
    except Exception:
        pass

    def vstatus(did):
        gs = val.get(did)
        if not gs:
            return "-"
        return "PASS" if all(p for _, p in gs) else "FAIL"

    out = []
    for i, d in enumerate(drows):
        import json
        try:
            pp = json.loads(d.get("physical_params") or "{}")
        except Exception:
            pp = {}
        try:
            cv = json.loads(d.get("constraint_values") or "{}")
        except Exception:
            cv = {}
        out.append({
            "id": d.get("id"), "iter": d.get("iter"),
            "mission_fom": r3(d.get("mission_fom")),
            "fom": r3(d.get("fom")),
            "LWL_m": r3(pp.get("LWL") or d.get("LWL")),
            "BWL_m": r3(pp.get("BWL")),
            "BWL_measured_m": r3(cv.get("BWL_measured") or pp.get("BWL")),
            "T_canoe_m": r3(pp.get("T_canoe")),
            "D_keel_m": r3(pp.get("D_keel")),
            "keel_chord_m": r3(pp.get("keel_chord")),
            "bulb_L": r1((pp.get("bulb_vol") or 0) * 1000),
            "ballast_frac": r3(pp.get("ballast_frac")),
            "T_over_L": r3(d.get("T_over_L")),
            "LDR": r1((d.get("LDR") if d.get("LDR") is not None else
                        (json.loads(d.get("constraint_values") or "{}").get("LDR")))),
            "GM_m": r3(d.get("gm")),
            "AVS_deg": r1(d.get("avs_deg")),
            "righting_energy_J": r1(d.get("righting_energy")),
            "heavy_leeway_deg": r1(d.get("heavy_leeway_deg")),
            "helm_comb_deg": r1(d.get("helm_combined_deg")),
            "mission_drive": r3(d.get("mission_drive")),
            "gust_margin": r3(d.get("gust_margin")),
            "draft_logistics": r3(d.get("draft_logistics_cost")),
            "Rt_N": r1(d.get("rt_total")),
            "validation": vstatus(d.get("id")),
            "best": "YES" if i == 0 else "",
        })
    with open("output/results_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    print(f"Wrote output/results_clean.csv ({len(out)} feasible designs)")
    if out:
        b = out[0]
        print(f"Best: id={b['id']} mission_fom={b['mission_fom']} "
              f"D_keel={b['D_keel_m']}m leeway={b['heavy_leeway_deg']}deg")


if __name__ == "__main__":
    main()
