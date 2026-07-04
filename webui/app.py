import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webui")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"

app = FastAPI(title="Boat Optimizer Web UI")

jinja_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
    cache_size=50,
)

def render(template_name: str, request: Request, **context):
    template = jinja_env.get_template(template_name)
    html = template.render(request=request, **context)
    return HTMLResponse(html)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

opt_process: Optional[subprocess.Popen] = None
opt_start_time: Optional[float] = None


def _db_path():
    return OUTPUT_DIR / "optimization.db"


def _get_db():
    import sqlite3
    conn = sqlite3.connect(str(_db_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _log_path():
    return OUTPUT_DIR / "pipeline.log"


def _config_path():
    return PROJECT_DIR / "config.yaml"


# ── Pages ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return render("dashboard.html", request)


@app.get("/hull", response_class=HTMLResponse)
async def hull_viewer(request: Request):
    return render("hull_viewer.html", request)


@app.get("/specs", response_class=HTMLResponse)
async def specs_page(request: Request):
    return render("specs.html", request)


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    return render("config.html", request)


@app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    return render("logs.html", request)


@app.get("/dryrun", response_class=HTMLResponse)
async def dryrun_page(request: Request):
    return render("dryrun.html", request)


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return render("results.html", request)


# ── API: Status ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    db_path = _db_path()
    if not db_path.exists():
        return {"total": 0, "feasible": 0, "best_fom": None, "current_iter": 0}

    conn = None
    try:
        conn = _get_db()
        cur = conn.execute("SELECT COUNT(*) FROM designs")
        total = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM designs WHERE feasible = 1")
        feasible = cur.fetchone()[0]
        cur = conn.execute("SELECT MAX(iter) FROM designs")
        max_iter = cur.fetchone()[0] or 0
        cur = conn.execute(
            "SELECT fom, rt_total, design_vector FROM designs WHERE feasible = 1 ORDER BY fom DESC LIMIT 1"
        )
        best = cur.fetchone()

        result = {
            "total": total,
            "feasible": feasible,
            "current_iter": max_iter,
            "best_fom": best[0] if best else None,
            "best_rt": best[1] if best else None,
        }
        if opt_start_time is not None:
            result["elapsed_s"] = int(time.time() - opt_start_time)
        result["running"] = opt_process is not None and opt_process.poll() is None
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


# ── API: Designs ───────────────────────────────────────────────────────────

@app.get("/api/designs")
async def api_designs(limit: int = 50, feasible_only: bool = False):
    db_path = _db_path()
    if not db_path.exists():
        return []

    conn = None
    try:
        conn = _get_db()
        where = "WHERE feasible = 1" if feasible_only else ""
        rows = conn.execute(
            f"SELECT id, iter, design_vector, feasible, fom, rt_total, rt_wave, "
            f"rt_friction, stability_index, roll_period, peak_accel, "
            f"constraint_values, constraint_violations, error_code, cad_stl_path "
            f"FROM designs {where} ORDER BY fom DESC LIMIT ?",
            (limit,),
        ).fetchall()
        designs = []
        for r in rows:
            d = dict(r)
            if d["constraint_values"]:
                d["constraint_values"] = json.loads(d["constraint_values"])
            if d["constraint_violations"]:
                d["constraint_violations"] = json.loads(d["constraint_violations"])
            designs.append(d)
        return designs
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


@app.get("/api/designs/{design_id}")
async def api_design_detail(design_id: int):
    db_path = _db_path()
    if not db_path.exists():
        raise HTTPException(404, "No database")

    conn = None
    try:
        conn = _get_db()
        row = conn.execute("SELECT * FROM designs WHERE id = ?", (design_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"Design {design_id} not found")
        d = dict(row)
        for field in ("constraint_values", "constraint_violations"):
            if d.get(field):
                d[field] = json.loads(d[field])

        try:
            val_rows = conn.execute(
                "SELECT * FROM validation WHERE design_id = ?", (design_id,)
            ).fetchall()
            d["validation"] = [dict(v) for v in val_rows]
        except Exception:
            d["validation"] = []
        return d
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if conn:
            conn.close()


@app.get("/api/designs/{design_id}/stl")
async def api_design_stl(design_id: int):
    db_path = _db_path()
    if not db_path.exists():
        raise HTTPException(404, "No database")
    conn = None
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT cad_stl_path FROM designs WHERE id = ?", (design_id,)
        ).fetchone()
        if not row or not row[0]:
            raise HTTPException(404, "No STL path for this design")
        stl_path = Path(row[0])
        if not stl_path.exists():
            raise HTTPException(404, f"STL file not found at {stl_path}")
        return FileResponse(str(stl_path), media_type="application/sla")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if conn:
            conn.close()


# ── API: Config ────────────────────────────────────────────────────────────

@app.get("/api/config")
async def api_get_config():
    cfg_path = _config_path()
    if not cfg_path.exists():
        return PlainTextResponse("config.yaml not found", status_code=404)
    return PlainTextResponse(cfg_path.read_text())


class ConfigUpdate(BaseModel):
    yaml_content: str


@app.post("/api/config")
async def api_save_config(update: ConfigUpdate):
    cfg_path = _config_path()
    try:
        parsed = yaml.safe_load(update.yaml_content)
        if not isinstance(parsed, dict):
            raise ValueError("Invalid YAML: must be a dictionary")
        with open(cfg_path, "w") as f:
            f.write(update.yaml_content)
        return {"status": "ok", "message": "Config saved"}
    except Exception as e:
        raise HTTPException(400, f"Invalid config: {e}")


# ── API: Optimization Control ──────────────────────────────────────────────

@app.post("/api/optimization/start")
async def api_opt_start():
    global opt_process, opt_start_time
    if opt_process is not None and opt_process.poll() is None:
        raise HTTPException(400, "Optimization already running")
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        opt_start_time = time.time()
        opt_process = subprocess.Popen(
            [
                "bash", "-c",
                f"export PATH=\"{os.environ.get('PATH', '')}\" && "
                f"source /opt/openfoam2512/etc/bashrc 2>/dev/null && "
                f"cd \"{PROJECT_DIR}\" && python run_optimization.py"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        logger.info(f"Optimization started (PID={opt_process.pid})")
        return {"status": "started", "pid": opt_process.pid}
    except Exception:
        opt_process = None
        opt_start_time = None
        raise HTTPException(500, "Failed to start optimization process")


@app.post("/api/optimization/stop")
async def api_opt_stop():
    global opt_process, opt_start_time
    if opt_process is None or opt_process.poll() is not None:
        raise HTTPException(400, "No optimization running")
    try:
        os.killpg(os.getpgid(opt_process.pid), signal.SIGTERM)
        opt_process.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(opt_process.pid), signal.SIGKILL)
        except Exception:
            pass
    opt_process = None
    opt_start_time = None
    logger.info("Optimization stopped")
    return {"status": "stopped"}


@app.get("/api/optimization/status")
async def api_opt_status():
    global opt_process, opt_start_time
    running = opt_process is not None and opt_process.poll() is None
    result = {"running": running}
    if running:
        result["pid"] = opt_process.pid
        result["elapsed_s"] = int(time.time() - opt_start_time) if opt_start_time else 0
    return result


# ── API: Logs ──────────────────────────────────────────────────────────────

@app.get("/api/logs")
async def api_get_logs():
    log_path = _log_path()
    if not log_path.exists():
        return PlainTextResponse("")
    return PlainTextResponse(log_path.read_text())


# ── API: Logs SSE ─────────────────────────────────────────────────────────

@app.get("/api/logs/stream")
async def api_log_stream(request: Request):
    log_path = _log_path()

    async def event_stream():
        pos = 0
        try:
            if log_path.exists():
                pos = log_path.stat().st_size
        except OSError:
            pos = 0
        while True:
            if await request.is_disconnected():
                break
            try:
                if log_path.exists():
                    current_size = log_path.stat().st_size
                    if current_size > pos:
                        with open(log_path) as f:
                            f.seek(pos)
                            new_data = f.read()
                            pos = f.tell()
                        for line in new_data.splitlines():
                            if line.strip():
                                yield f"data: {json.dumps({'line': line})}\n\n"
            except (OSError, IOError):
                pos = 0
                await asyncio.sleep(1)
            await asyncio.sleep(0.5)
        yield "data: {\"line\": \"[stream ended]\"}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── API: Dry Run ───────────────────────────────────────────────────────────

@app.post("/api/dryrun")
async def api_dryrun():
    try:
        result = subprocess.run(
            [
                "bash", "-c",
                f"export PATH=\"{os.environ.get('PATH', '')}\" && "
                f"source /opt/openfoam2512/etc/bashrc 2>/dev/null && "
                f"cd \"{PROJECT_DIR}\" && python run_optimization.py --dry-run"
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "status": "ok",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": "Dry-run timed out after 300s"}
    except Exception as e:
        return {"status": "error", "stdout": "", "stderr": str(e)}


# ── API: DB Stats ──────────────────────────────────────────────────────────

@app.get("/api/db-stats")
async def api_db_stats():
    db_path = _db_path()
    if not db_path.exists():
        return {"total": 0}

    conn = None
    try:
        conn = _get_db()

        def _count(table, where=""):
            try:
                return conn.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
            except Exception:
                return 0

        total = _count("designs")
        feasible = _count("designs", "WHERE feasible = 1")
        calibrations = _count("calibration")
        validations = _count("validation")

        # constraint violation frequency
        cur = conn.execute(
            "SELECT constraint_violations FROM designs WHERE feasible = 0 AND constraint_violations IS NOT NULL"
        )
        violation_counts = {}
        for row in cur.fetchall():
            try:
                viols = json.loads(row[0])
                for v in viols:
                    key = v.split("=")[0] if "=" in v else v
                    violation_counts[key] = violation_counts.get(key, 0) + 1
            except Exception:
                pass

        return {
            "total": total,
            "feasible": feasible,
            "calibrations": calibrations,
            "validations": validations,
            "violation_counts": violation_counts,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()


# ── API: Plots ─────────────────────────────────────────────────────────────

@app.get("/api/plots/{filename:path}")
async def api_plot(filename: str):
    plot_dir = OUTPUT_DIR.resolve()
    candidates = [
        plot_dir / filename,
        plot_dir / "quick_test" / filename,
        plot_dir / "medium_test" / filename,
    ]
    for p in candidates:
        resolved = p.resolve()
        if not str(resolved).startswith(str(plot_dir)):
            raise HTTPException(403, "Access denied")
        if resolved.exists():
            return FileResponse(str(resolved), media_type="image/png")
    raise HTTPException(404, f"Plot {filename} not found")


# ── API: GZ Curve Data ─────────────────────────────────────────────────────

@app.get("/api/designs/{design_id}/gz")
async def api_design_gz(design_id: int):
    db_path = _db_path()
    if not db_path.exists():
        raise HTTPException(404, "No database")

    conn = None
    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT design_vector, cad_stl_path FROM designs WHERE id = ?",
            (design_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, f"Design {design_id} not found")

        stl_path = row["cad_stl_path"]
        if not stl_path or not Path(stl_path).exists():
            return {"error": "STL not available for GZ computation"}

        dv = json.loads(row["design_vector"])
        from hull_opt.hydrostatics import compute_gz_curve, compute_cg_z
        from hull_opt.config import design_vector_names

        x_dict = dict(zip(design_vector_names(), dv))
        cg_z = compute_cg_z(x_dict)
        gz = compute_gz_curve(stl_path, cg_z=cg_z, n_angles=37, max_heel=180.0)
        return {
            "angles": gz[:, 0].tolist(),
            "gz": gz[:, 1].tolist(),
            "volumes": gz[:, 2].tolist(),
            "cg_z": cg_z,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if conn:
            conn.close()


# ── API: Reset Database ─────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    confirmation: str

@app.post("/api/reset")
async def api_reset(req: ResetRequest):
    global opt_process, opt_start_time
    if req.confirmation != "yes":
        raise HTTPException(400, "Must confirm with 'yes'")

    db_path = _db_path()

    # Stop optimization if running
    if opt_process is not None and opt_process.poll() is None:
        try:
            os.killpg(os.getpgid(opt_process.pid), signal.SIGTERM)
            opt_process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(opt_process.pid), signal.SIGKILL)
            except Exception:
                pass
        opt_process = None
        opt_start_time = None

    # Delete database
    deleted = False
    if db_path.exists():
        try:
            db_path.unlink()
            deleted = True
        except Exception as e:
            raise HTTPException(500, f"Failed to delete database: {e}")

    logger.warning("Database reset complete")
    return {"status": "ok", "deleted": deleted}


# ── API: Calibration History ───────────────────────────────────────────────

@app.get("/api/calibrations")
async def api_calibrations():
    db_path = _db_path()
    if not db_path.exists():
        return []

    conn = None
    try:
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM calibration ORDER BY iter"
            ).fetchall()
            cals = [dict(r) for r in rows]
        except Exception:
            cals = []
        return cals
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()
