"""Rescore campaign designs under the mission FoM (Bug #169).

Loads designs from output/optimization.db, re-evaluates with current code,
prints old-FoM vs mission-FoM with component breakdown. Usage:
    venv/bin/python scripts/rescore_mission.py [--write] [id ...]
--write persists mission_fom + breakdown to the DB (campaign fom untouched,
fixing the "CSV lags reality" gap). Without --write: audit only, no DB writes.
"""
import json
import sqlite3
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")
import json
import sqlite3
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

IDS_DEFAULT = [105, 149, 134, 139, 151, 153, 159, 130, 59, 131, 141, 98]


def main():
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    config = load_config("config.yaml")
    ids = [int(a) for a in args] or IDS_DEFAULT
    db = None
    if write:
        from hull_opt.database import OptimizationDatabase
        db = OptimizationDatabase(config.paths.database)
    c = sqlite3.connect("output/optimization.db")
    rows = []
    for did in ids:
        r = c.execute(
            "select id, iter, fom, feasible, design_vector from designs where id=?",
            (did,)).fetchone()
        if not r:
            print(did, "MISSING")
            continue
        dv = np.array(json.loads(r[4]), dtype=np.float64)
        td = tempfile.mkdtemp(prefix=f"rescore_{did}_")
        try:
            res = evaluate_low_fidelity(dv, config, output_dir=td,
                                        drag_factor=1.0, iteration=0)
        except Exception as e:  # noqa: BLE001 — audit must not die on one design
            print(f"id={did} iter={r[1]} EVAL-ERROR {e}")
            continue
        rows.append({
            "id": did, "iter": r[1], "old_fom": r[2],
            "feasible": res.feasible, "mission_fom": res.fom,
            "err": res.error_code,
            "rt_ocean": getattr(res, "rt_ocean", None),
            "mission_drive": getattr(res, "mission_drive", None),
            "gust_margin": getattr(res, "gust_margin", None),
            "heavy_leeway": getattr(res, "heavy_leeway_deg", None),
            "logistics": getattr(res, "draft_logistics_cost", None),
            "stability": res.stability_index,
        })
        if write and db is not None and res.error_code is None:
            db.update_design_mission(did, {
                "mission_fom": res.fom,
                "mission_drive": getattr(res, "mission_drive", None),
                "gust_margin": getattr(res, "gust_margin", None),
                "heavy_leeway_deg": getattr(res, "heavy_leeway_deg", None),
                "draft_logistics_cost": getattr(res, "draft_logistics_cost", None),
            })
            print(f"  wrote mission_fom={res.fom:.3f} to DB id={did}")
    rows.sort(key=lambda d: (d["mission_fom"] is None, -(d["mission_fom"] or -1e18)))
    print(f"{'id':>5} {'iter':>5} {'old':>8} {'mission':>8} {'feas':>5} "
          f"{'rt_oc':>7} {'drive':>6} {'gust':>6} {'leeway':>6} {'logist':>6} {'stab':>5} err")
    for d in rows:
        print(f"{d['id']:>5} {d['iter']:>5} {d['old_fom']:>8.3f} "
              f"{(d['mission_fom'] if d['mission_fom'] is not None else float('nan')):>8.3f} "
              f"{str(d['feasible']):>5} "
              f"{(d['rt_ocean'] or float('nan')):>7.1f} "
              f"{(d['mission_drive'] or 0):>6.3f} "
              f"{(d['gust_margin'] if d['gust_margin'] is not None else float('nan')):>6.2f} "
              f"{(d['heavy_leeway'] if d['heavy_leeway'] is not None else float('nan')):>6.1f} "
              f"{(d['logistics'] or 0):>6.3f} {d['stability']:>5.2f} {d['err']}")


if __name__ == "__main__":
    main()
