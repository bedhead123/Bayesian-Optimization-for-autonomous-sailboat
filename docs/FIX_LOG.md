# Fix Log

## BUG-001: Duplicate keel root closure faces
- **File**: `hull_opt/geometry.py:1154-1160`
- **Fix**: Removed the first duplicate root closure block (`for j in range(n_chord - 1)` with `all_faces.append([a, sa, b])` and `all_faces.append([b, sa, sb])`). The root closure at lines 1176-1182 (now 1168-1175) is sufficient.

## BUG-002/003: Bulb volume ignored in CG_z when ballast_frac > 0.01
- **File**: `hull_opt/geometry.py:1360-1376`
- **File**: `hull_opt/hydrostatics.py:197-224`
- **Fix**: Changed CG_z computation so that:
  - `bulb_mass` always comes from actual bulb geometry (`bulb_vol * 11340`)
  - `keel_mass` always comes from actual keel geometry
  - `ballast_mass` is computed as `total_mass * ballast_frac` (separate additional mass)
  - `hull_mass = total_mass - bulb_mass - keel_mass - ballast_mass`, clamped to 0 if negative
  - `ballast_cg_z = -(T_hull + D_keel * 0.5)` (ballast distributed in keel region)
  - CG includes `ballast_mass * ballast_cg_z` term

## BUG-004: Convexity threshold mismatch
- **File**: `hull_opt/geometry_validator.py:75`
- **Fix**: Changed `convexity < 0.25` to `convexity < 0.35` to match `geometry.py:334`.

## BUG-005: SAC clip upper bound mismatch
- **File**: `hull_opt/geometry.py:845`
- **Fix**: Changed `np.clip(sac_avg_scale, 0.1, 5.0)` to `np.clip(sac_avg_scale, 0.1, 2.5)`.

## BUG-006: Exception swallowing in _validate_hull_mesh self-intersection check
- **File**: `hull_opt/geometry.py:45-46`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: return False, f"Self-intersection check error: {e}"`.

## BUG-007: Exception swallowing in _validate_hull_mesh spike detection
- **File**: `hull_opt/geometry.py:72-73`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: logger.debug(f"Spike detection error in _validate_hull_mesh: {e}")`.

## BUG-008: Exception swallowing in _validate_hull_mesh sliver triangle check
- **File**: `hull_opt/geometry.py:100-101`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: logger.debug(f"Sliver triangle check error: {e}")`.

## BUG-009: Exception swallowing in _validate_hull_mesh normal consistency check
- **File**: `hull_opt/geometry.py:117-119`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: logger.debug(...); return False, f"Normal consistency check failed: {e}"`.

## BUG-010: Exception swallowing in geometry_validator.py convexity check
- **File**: `hull_opt/geometry_validator.py:79-80`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: return False, f"Convexity check error: {e}"`.

## BUG-011: Exception swallowing in _check_mesh_spikes
- **File**: `hull_opt/geometry.py:282-316`
- **Fix**: Removed the try/except entirely. The calling code in `generate_hull` handles exceptions; the function now raises naturally.

## BUG-012: Exception swallowing in keel-hull penetration check
- **File**: `hull_opt/geometry.py:494-495`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: logger.warning(f"Penetration check failed for body {j}: {e}")`.

## BUG-013: Half-width station sampling too narrow
- **File**: `hull_opt/geometry.py:435`
- **File**: `hull_opt/constraints.py:33`
- **Fix**: Changed `station_width = (x_vals[1] - x_vals[0]) * 0.5` to `* 1.0` in both files.

## BUG-014: Three independent CG_z computations
- **File**: `hull_opt/low_fidelity.py:114`
- **File**: `hull_opt/low_fidelity.py:307`
- **Fix**: Changed `cg_z = compute_cg_z(...)` to `cg_z = hydro.get("cg_z", compute_cg_z(...))` in both locations. This prefers the CG_z already computed by the hydrostatics module, ensuring consistency.

## BUG-018: Exception swallowing in geometry_validator.py self-intersection check
- **File**: `hull_opt/geometry_validator.py:102-103`
- **Fix**: Changed `except Exception: pass` to `except Exception as e: return False, f"Self-intersection check error: {e}"`.

## BUG-019: Non-watertight combined mesh not raising error
- **File**: `hull_opt/geometry.py:1037-1040`
- **Fix**: Changed from `logger.warning` to `hull_mesh.fill_holes()` followed by `raise ValueError(...)` if still not watertight.

## BUG-020: Spike detection threshold too lenient (179°)
- **File**: `hull_opt/geometry.py:70`
- **Fix**: Changed `np.deg2rad(179)` to `np.deg2rad(150)`.

## ADDITIONAL FIXES

### geometry_validator.py exception swallowing (normal check)
- **File**: `hull_opt/geometry_validator.py:119-120`
- **Fix**: Changed to `except Exception as e: return False, f"Normal check error: {e}"`.

### geometry_validator.py exception swallowing (edge ratio)
- **File**: `hull_opt/geometry_validator.py:135-136`
- **Fix**: Changed to `except Exception as e: return False, f"Edge ratio check error: {e}"`.

### geometry_validator.py exception swallowing (sliver triangle)
- **File**: `hull_opt/geometry_validator.py:152-153`
- **Fix**: Changed to `except Exception as e: return False, f"Sliver triangle check error: {e}"`.

### hydrostatics.py: Waterplane properties exception logging
- **File**: `hull_opt/hydrostatics.py:105-106`
- **Fix**: Added `logger.debug(...)` to log the exception before returning defaults.

### hydrostatics.py: GZ curve slice exception logging
- **File**: `hull_opt/hydrostatics.py:148-149`
- **Fix**: Added `logger.debug(...)` to log the exception before setting GZ to 0.

### hydrostatics.py: Added logger
- **File**: `hull_opt/hydrostatics.py:4,8`
- **Fix**: Added `import logging` and `logger = logging.getLogger(__name__)`.

### low_fidelity.py: Peak acceleration fallback value
- **File**: `hull_opt/low_fidelity.py:415-416`
- **Fix**: Changed `return 10.0` (too permissive) to `return 60.0` (conservative penalty).
