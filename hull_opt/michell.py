"""
Michell integral wave resistance for thin ships.
Uses sigma-transformation (σ = 1/cos θ) to avoid the cos³θ singularity.
Key exports: compute_wave_resistance_michell()
Bugs fixed: missing g² factor in prefactor (#1), zero-speed guard (#6)
"""
import numpy as np


def compute_wave_resistance_michell(half_breadth_func, LWL: float, B: float,
                                    T: float, speed_ms: float,
                                    rho: float = 1025.0, g: float = 9.81,
                                    n_theta: int = 60) -> float:
    if speed_ms <= 0:
        return 0.0
    k0 = g / max(1e-10, speed_ms ** 2)

    # use sigma = 1/cos(theta) as integration variable
    # this transforms the integral to avoid the cos^3 singularity
    sigma_max = 4.0
    # Use uniform θ-spacing to cluster points near σ=1 singularity.
    # Starting at θ=half-interval guarantees the first trapezoid bisects
    # the leading edge of the integrable singularity and avoids the
    # numerical spike at σ≈1+5e-13 that a fixed 1e-6 offset would cause.
    theta_max = np.arccos(1.0 / sigma_max)
    theta = np.linspace(theta_max / (2.0 * n_theta), theta_max, n_theta)
    sigma = 1.0 / np.cos(theta)

    n_x = 40
    n_z = 20
    x = np.linspace(-LWL / 2, LWL / 2, n_x)
    z = np.linspace(-T, 0, n_z)
    xx, zz = np.meshgrid(x, z)

    f = half_breadth_func(xx, zz)
    if np.isscalar(f):
        f = np.full_like(xx, f)
    dx = x[1] - x[0]
    df_dx = np.gradient(f, dx, axis=1)

    # Clip extreme gradients to prevent numerical gaming at bow/stern
    max_grad = 10.0
    df_dx = np.clip(df_dx, -max_grad, max_grad)

    integrand = np.zeros(n_theta)
    for i in range(n_theta):
        sig = sigma[i]
        kc = k0 * sig
        kz_exp = k0 * sig ** 2

        exp_z = np.exp(kz_exp * zz)
        cos_x = np.cos(kc * xx)
        sin_x = np.sin(kc * xx)

        I_val = np.trapezoid(np.trapezoid(df_dx * exp_z * cos_x, x, axis=1), z)
        J_val = np.trapezoid(np.trapezoid(df_dx * exp_z * sin_x, x, axis=1), z)

        integrand[i] = (I_val ** 2 + J_val ** 2) * sig ** 2 / \
                       max(1e-10, np.sqrt(sig ** 2 - 1.0))

    Rw = (4.0 * rho * g ** 2) / (np.pi * speed_ms ** 2) * \
         np.trapezoid(integrand, sigma)

    # Convergence check: re-evaluate at 2x resolution to verify stability
    n_x2 = n_x * 2
    n_z2 = n_z * 2
    x2 = np.linspace(-LWL / 2, LWL / 2, n_x2)
    z2 = np.linspace(-T, 0, n_z2)
    xx2, zz2 = np.meshgrid(x2, z2)
    f2 = half_breadth_func(xx2, zz2)
    if np.isscalar(f2):
        f2 = np.full_like(xx2, f2)
    dx2 = x2[1] - x2[0]
    df_dx2 = np.clip(np.gradient(f2, dx2, axis=1), -max_grad, max_grad)

    integrand2 = np.zeros(n_theta)
    for i in range(n_theta):
        sig = sigma[i]
        kc = k0 * sig
        kz_exp = k0 * sig ** 2

        exp_z2 = np.exp(kz_exp * zz2)
        cos_x2 = np.cos(kc * xx2)
        sin_x2 = np.sin(kc * xx2)

        I_val = np.trapezoid(np.trapezoid(df_dx2 * exp_z2 * cos_x2, x2, axis=1), z2)
        J_val = np.trapezoid(np.trapezoid(df_dx2 * exp_z2 * sin_x2, x2, axis=1), z2)
        integrand2[i] = (I_val ** 2 + J_val ** 2) * sig ** 2 / \
                        max(1e-10, np.sqrt(sig ** 2 - 1.0))
    Rw2 = (4.0 * rho * g ** 2) / (np.pi * speed_ms ** 2) * \
          np.trapezoid(integrand2, sigma)

    if Rw > 0 and Rw2 > 0:
        ratio = max(Rw, Rw2) / min(Rw, Rw2)
        if ratio > 1.5:
            import warnings
            warnings.warn(f"Michell integral not converged: {Rw:.6f} vs {Rw2:.6f} at 2x resolution (ratio={ratio:.3f})")
            Rw = (Rw + Rw2) * 0.5

    return max(0.0, Rw)
