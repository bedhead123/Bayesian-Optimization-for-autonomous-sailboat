"""
ITTC-1957 friction line and total resistance (friction + wave).
Key exports: compute_frictional_resistance(), compute_total_resistance()
"""
import numpy as np


def compute_frictional_resistance(speed_ms: float, wetted_area: float,
                                  LWL: float, rho: float = 1025.0,
                                  nu: float = 1.19e-6,
                                  form_factor: float = 0.1,
                                  appendage_area: float = 0.0,
                                  roughness_mult: float = 1.0) -> float:
    Re = speed_ms * LWL / max(1e-10, nu)
    if Re < 1e3:
        return 0.0
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2
    wetted = wetted_area + appendage_area
    Rf = 0.5 * rho * speed_ms ** 2 * wetted * Cf * (1.0 + form_factor)
    return Rf * roughness_mult


def compute_total_resistance(speed_ms: float, wetted_area: float,
                             LWL: float, rho: float = 1025.0,
                             nu: float = 1.19e-6,
                             form_factor: float = 0.1,
                              wave_resistance: float = 0.0,
                             appendage_area: float = 0.0,
                             roughness_mult: float = 1.0) -> tuple[float, float, float]:
    Rf = compute_frictional_resistance(speed_ms, wetted_area, LWL, rho, nu, form_factor,
                                       appendage_area=appendage_area,
                                       roughness_mult=roughness_mult)
    Rt = Rf + wave_resistance
    return Rt, Rf, wave_resistance
