"""Replay production design vectors through the real BEM pipeline.

The production run evaluated 320 designs with LHS sampling on bem_skip=True
(fake roll_period/peak_accel) and BO designs failing with E_RAO decimation
errors under a broken environment. This script re-runs every design through
``evaluate_low_fidelity(..., bem_skip=False)`` under the repo venv and writes
a results CSV with the real RAO metrics.

Usage:
    venv/bin/python scripts/replay_bem.py                    # all 320 from output/optimization.db
    venv/bin/python scripts/replay_bem.py --lhs-only         # LHS rows (iter < lhs_max) only
    venv/bin/python scripts/replay_bem.py --limit 5 --workers 2
    venv/bin/python scripts/replay_bem.py --db /path/to.db --csv /path/to/results.csv
"""
import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = REPO_ROOT / "output" / "optimization.db"
DEFAULT_CSV = REPO_ROOT / "output" / "results.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "replay_bem_results.csv"
REPLAY_DIR = "replay_bem"

REPORT_COLUMNS = ["id", "iter_num", "feasible", "fom", "roll_period",
                  "peak_accel", "gm", "error_code", "rt_total", "bem_ok"]

_PARAM_ORDER = ["LWL", "BWL", "T_canoe", "Cp", "Cm", "LCB", "D_keel",
                "keel_chord", "bulb_vol", "bulb_pos", "E", "flare",
                "deadrise", "bilge_r", "keel_rake", "ballast_frac",
                "wingsail_pos"]

PROGRESS_EVERY = 25


def check_venv() -> bool:
    """Refuse to run outside the repo venv: Capytaine/decimation deps
    (fast_simplification) are only installed there."""
    if "venv" in sys.executable or "virtualenv" in sys.prefix:
        return True
    print(f"ERROR: replay_bem.py must run under the repo venv, not "
          f"{sys.executable} (sys.prefix={sys.prefix}).",
          file=sys.stderr)
    print(f"Use: {REPO_ROOT / 'venv' / 'bin' / 'python'} "
          f"{Path(__file__).resolve()}", file=sys.stderr)
    return False


def lhs_iter_bound(config) -> int:
    """LHS rows occupy iters [0, lhs_max); surrogate._initial_sampling
    evaluates exactly min(lhs_max, n_initial) designs with bem_skip=True
    starting at iteration 0, and BO continues from that count."""
    lhs_max = getattr(config.optimization, "lhs_max", 40)
    return min(int(lhs_max), int(config.optimization.n_initial))


def is_lhs_row(row: dict, lhs_bound: int) -> bool:
    return row["iter"] < lhs_bound


def load_designs_from_db(db_path: Path, lhs_only: bool,
                         lhs_bound: int) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, iter, design_vector, feasible, fom, roll_period, "
            "peak_accel, error_code, rt_total, drag_factor "
            "FROM designs ORDER BY iter"
        ).fetchall()
    finally:
        conn.close()
    designs = []
    for r in rows:
        if lhs_only and not is_lhs_row(dict(r), lhs_bound):
            continue
        try:
            dv = np.asarray(json.loads(r["design_vector"]), dtype=np.float64)
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            dv = None
            err = f"E_DV_LOAD:{e}"
        else:
            err = None
        designs.append({
            "id": r["id"], "iter": r["iter"],
            "design_vector": dv,
            "load_error": err,
            "feasible": bool(r["feasible"]),
            "fom": r["fom"], "roll_period": r["roll_period"],
            "peak_accel": r["peak_accel"], "error_code": r["error_code"],
            "rt_total": r["rt_total"],
            "drag_factor": float(r["drag_factor"]) if r["drag_factor"] is not None else 1.0,
        })
    return designs


def _bulb_vol_max_for(keel_chord: float) -> float:
    """Geometric max bulb volume, mirroring param_layer.bulb_vol_max_for."""
    return 4.0 / 3.0 * np.pi * (keel_chord * 0.5) ** 3


def physical_to_raw(pp: dict, config) -> np.ndarray:
    """Invert design_vector_to_physical (param_layer.py) per-dimension.

    Each raw element r_i maps to x_i = lo + (hi - lo) * sigmoid(r_i), so the
    inverse is r_i = logit((x_i - lo) / (hi - lo)). bulb_vol's effective upper
    bound is capped by keel_chord (param_layer.py: bulb_vol_max_for), so its
    inverse must apply the same cap. Used only for CSV sources, which store
    decoded physical params instead of raw vectors.
    """
    b = config.bounds

    def _get(name):
        v = pp.get(name)
        if v is None:
            raise ValueError(f"CSV row missing physical param '{name}'")
        v = float(v)
        if not np.isfinite(v):
            raise ValueError(f"CSV physical param '{name}' is not finite: {v}")
        return v

    def _logit(v, lo, hi):
        p = np.clip((v - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    raw = []
    for name in _PARAM_ORDER:
        lo, hi = getattr(b, name)
        v = _get(name)
        if name == "bulb_vol":
            kc = _get("keel_chord")
            hi = min(hi, _bulb_vol_max_for(kc))
        raw.append(_logit(v, lo, hi))
    return np.asarray(raw, dtype=np.float64)


def load_designs_from_csv(csv_path: Path, lhs_only: bool,
                          lhs_bound: int, config) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "iter" not in reader.fieldnames:
            raise ValueError(
                f"CSV {csv_path} has no 'iter' column; expected the "
                f"results.csv layout from run_optimization.py "
                f"generate_results_report"
            )
        for r in reader:
            rows.append(r)
    designs = []
    for r in rows:
        try:
            iter_num = int(float(r["iter"]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"CSV row has non-integer 'iter': {r.get('iter')!r}") from e
        if lhs_only and iter_num >= lhs_bound:
            continue
        try:
            dv = physical_to_raw(r, config)
        except ValueError as e:
            dv = None
            err = f"E_DV_LOAD:{e}"
        else:
            err = None
        def _num(v):
            try:
                return float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None
        designs.append({
            "id": None, "iter": iter_num,
            "design_vector": dv,
            "load_error": err,
            "feasible": bool(r.get("feasible") == "yes"),
            "fom": _num(r.get("fom")),
            "roll_period": _num(r.get("roll_period")),
            "peak_accel": _num(r.get("peak_accel")),
            "error_code": r.get("error_code") or None,
            "rt_total": _num(r.get("rt_total")),
            "drag_factor": 1.0,
        })
    return designs


def load_designs(db_path: Optional[Path], csv_path: Optional[Path],
                 lhs_only: bool, lhs_bound: int, config) -> tuple[list[dict], str]:
    if db_path is None and csv_path is None:
        if DEFAULT_DB.exists() and DEFAULT_DB.stat().st_size > 0:
            db_path = DEFAULT_DB
        elif DEFAULT_CSV.exists():
            csv_path = DEFAULT_CSV
        else:
            print(f"ERROR: no data source found — expected DB "
                  f"{DEFAULT_DB} or CSV {DEFAULT_CSV}; pass --db/--csv.",
                  file=sys.stderr)
            sys.exit(1)
    if db_path is not None:
        designs = load_designs_from_db(Path(db_path), lhs_only, lhs_bound)
        return designs, f"DB {db_path}"
    designs = load_designs_from_csv(Path(csv_path), lhs_only, lhs_bound, config)
    return designs, f"CSV {csv_path}"


def result_to_row(res, design: dict) -> dict:
    cv = getattr(res, "constraint_values", None) or {}
    gm = getattr(res, "gm", None)
    if gm is None:
        gm = cv.get("gm")
    error_code = getattr(res, "error_code", None)
    if error_code is None and design.get("load_error"):
        error_code = design["load_error"]
    return {
        "id": design.get("id"),
        "iter_num": design["iter"],
        "feasible": bool(getattr(res, "feasible", False)),
        "fom": getattr(res, "fom", None),
        "roll_period": getattr(res, "roll_period", 0.0),
        "peak_accel": getattr(res, "peak_accel", 0.0),
        "gm": gm,
        "error_code": error_code,
        "rt_total": getattr(res, "rt_total", None),
    }


def is_bem_ok(res) -> bool:
    return getattr(res, "error_code", None) is None and \
        (getattr(res, "roll_period", 0.0) or 0.0) > 0.0


def _replay_worker(dv, config, out_dir, drag_factor, iteration):
    """Replicates surrogate._evaluate_one_wrapper with bem_skip=False."""
    from threadpoolctl import threadpool_limits
    n_threads = int(os.environ.get("BOAT_BLAS_THREADS", "4"))
    with threadpool_limits(limits=n_threads):
        from hull_opt.low_fidelity import evaluate_low_fidelity
        return evaluate_low_fidelity(
            dv, config,
            output_dir=out_dir,
            drag_factor=drag_factor,
            iteration=iteration,
            bem_skip=False,
        )


def run_replay(designs: list[dict], config, out_dir: str,
               workers: int = 1) -> list[dict]:
    total = len(designs)
    out_rows = []
    done = 0
    t0 = time.time()

    def _log_progress():
        if done % PROGRESS_EVERY == 0 or done == total:
            sys.stderr.write(
                f"[replay_bem] {done}/{total} designs evaluated "
                f"({time.time() - t0:.1f}s)\n")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_replay_worker, d["design_vector"], config,
                            out_dir, d["drag_factor"], d["iter"]): d
                for d in designs
            }
            for f in as_completed(futures):
                d = futures[f]
                try:
                    res = f.result()
                except Exception as e:
                    out_rows.append(result_to_row(
                        _error_result(f"E_REPLAY:{e}"), d))
                else:
                    out_rows.append(result_to_row(res, d))
                done += 1
                _log_progress()
    else:
        for d in designs:
            if d["design_vector"] is None:
                out_rows.append(result_to_row(_error_result(
                    d.get("load_error") or "E_REPLAY:no design vector"), d))
                done += 1
                _log_progress()
                continue
            t1 = time.time()
            try:
                res = _replay_worker(d["design_vector"], config, out_dir,
                                     d["drag_factor"], d["iter"])
            except Exception as e:
                res = _error_result(f"E_REPLAY:{e}")
            out_rows.append(result_to_row(res, d))
            done += 1
            _log_progress()
    return out_rows


def _error_result(msg: str):
    res = type("ReplayError", (), {})()
    res.error_code = str(msg)
    res.feasible = False
    res.fom = None
    res.roll_period = 0.0
    res.peak_accel = 0.0
    res.gm = None
    res.rt_total = None
    res.constraint_values = {}
    return res


def write_results_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k])
                             for k in REPORT_COLUMNS})


def summarize(rows: list[dict], designs: list[dict], lhs_only: bool) -> None:
    n = len(rows)
    n_ok = sum(1 for r in rows if r["bem_ok"])
    n_err = sum(1 for r in rows if r["error_code"])
    errs = Counter((r["error_code"].split(":", 1)[0] if r["error_code"] else "ok")
                   for r in rows)
    top = [k for k in errs if k != "ok"]
    print("\nReplay summary:")
    print(f"  N evaluated: {n}")
    print(f"  N with bem_ok: {n_ok}")
    print(f"  N with error: {n_err}")
    print("  Top error codes:")
    if top:
        for code, cnt in errs.most_common(3):
            if code != "ok":
                print(f"    {code}: {cnt}")
    else:
        print("    (none)")
    if lhs_only:
        before = {d["iter"]: d["roll_period"] for d in designs}
        pairs = [(r["iter_num"], before.get(r["iter_num"]), r["roll_period"])
                 for r in rows if r["iter_num"] in before]
        print("  LHS roll_period before -> after (first 5):")
        for i, b, a in pairs[:5]:
            b_s = "None" if b is None else f"{b:.4f}"
            a_s = "None" if a is None else f"{a:.4f}"
            print(f"    iter {i}: {b_s} -> {a_s}")
    print("")


def main() -> int:
    if not check_venv():
        return 2

    parser = argparse.ArgumentParser(
        description="Replay stored design vectors through real BEM "
                    "(bem_skip=False) and write replay results.")
    parser.add_argument("--db", default=None,
                        help=f"SQLite DB path (default: {DEFAULT_DB} if present)")
    parser.add_argument("--csv", default=None,
                        help=f"results.csv path (default: {DEFAULT_CSV} if "
                             f"no DB present)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Replay only the first N designs (0 = all)")
    parser.add_argument("--lhs-only", action="store_true",
                        help="Only replay LHS rows (iter < lhs_max)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Results CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--config", default="config.yaml",
                        help="Config YAML (default: config.yaml)")
    args = parser.parse_args()

    from hull_opt.config import load_config
    config = load_config(str(REPO_ROOT / args.config))
    lhs_bound = lhs_iter_bound(config)

    designs, source = load_designs(
        Path(args.db) if args.db else None,
        Path(args.csv) if args.csv else None,
        args.lhs_only, lhs_bound, config,
    )
    if not designs:
        print(f"ERROR: no designs found in {source}; nothing to replay.",
              file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        designs = designs[:args.limit]

    print(f"Replaying {len(designs)} designs from {source} "
          f"(bem_skip=False, workers={args.workers}); "
          f"LHS bound = iter < {lhs_bound}")
    out_dir = str(REPO_ROOT / "output" / REPLAY_DIR)
    rows = run_replay(designs, config, out_dir, workers=args.workers)

    for r in rows:
        r["bem_ok"] = r["error_code"] is None and (r["roll_period"] or 0.0) > 0.0

    out_path = Path(args.out)
    write_results_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")

    summarize(rows, designs, args.lhs_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
