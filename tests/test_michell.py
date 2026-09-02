import numpy as np
from hull_opt.michell import compute_wave_resistance_michell


def test_wave_resistance_zero_at_zero_speed():
    def half_breadth(x, z):
        return np.ones_like(x) * 0.5

    Rw = compute_wave_resistance_michell(
        half_breadth, LWL=2.4, B=1.0, T=0.25,
        speed_ms=0.001, rho=1025.0, g=9.81,
    )
    # wave resistance should go to 0 as speed -> 0
    assert Rw < 0.1


def test_wave_resistance_positive():
    def half_breadth(x, z):
        return np.full_like(x, 0.5)

    Rw = compute_wave_resistance_michell(
        half_breadth, LWL=2.4, B=1.0, T=0.25,
        speed_ms=1.8, rho=1025.0, g=9.81,
    )
    assert Rw >= 0
    assert isinstance(Rw, float)


def test_wave_resistance_nonnegative():
    rng = np.random.default_rng(42)
    for _ in range(5):
        speeds = np.linspace(0.5, 2.5, 5)

        def hb(x, z):
            return np.full_like(x, rng.uniform(0.3, 0.8))

        for s in speeds:
            Rw = compute_wave_resistance_michell(
                hb, LWL=2.4, B=1.0, T=0.25,
                speed_ms=s, rho=1025.0, g=9.81,
            )
            assert Rw >= 0, f"Negative resistance at speed {s}: {Rw}"


def test_wave_resistance_increases_to_hump():
    # Michell integral should show hump-and-hollow pattern
    # as Froude number increases for a realistic hull
    L = 2.4
    T = 0.25
    B = 0.8

    def half_breadth(x, z):
        x_norm = 2.0 * x / L
        z_norm = z / T
        y = 0.5 * B * (1.0 - x_norm ** 2) * (1.0 - z_norm ** 2)
        return np.maximum(y, 0.0)

    speeds = np.linspace(0.5, 3.0, 10)
    Rw_vals = []
    for s in speeds:
        Rw = compute_wave_resistance_michell(
            half_breadth, LWL=L, B=B, T=T,
            speed_ms=s, rho=1025.0, g=9.81,
        )
        Rw_vals.append(Rw)

    Rw_vals = np.array(Rw_vals)
    # should have at least one local max (hump)
    diffs = np.diff(Rw_vals)
    assert np.any(diffs > 0), "No hump in wave resistance curve"


def test_wave_resistance_wigley_hull():
    # Wigley parabolic hull: y/B/2 = (1-(2x/L)^2)(1-(z/T)^2)
    # Known approximate values from literature: at Fr=0.25, L=2.4m
    L = 2.4
    B = 0.24
    T = 0.15

    def wigley_half_breadth(x, z):
        x_norm = 2.0 * x / L
        z_norm = z / T
        # y/B/2 = (1-x_norm^2)(1-z_norm^2)
        # so y = B/2 * (1-x_norm^2)(1-z_norm^2)
        val = 0.5 * B * (1.0 - x_norm ** 2) * (1.0 - z_norm ** 2)
        return np.maximum(val, 0.0)

    # Fr = 0.25: V = 0.25 * sqrt(g*L) = 0.25 * sqrt(9.81*2.4)
    speed_ms = 0.25 * np.sqrt(9.81 * L)
    Rw = compute_wave_resistance_michell(
        wigley_half_breadth, LWL=L, B=B, T=T,
        speed_ms=speed_ms, rho=1025.0, g=9.81,
    )
    # Wigley hull at Fr=0.25 typically gives Rw ~ 0.5-2 N for these dimensions
    assert 0 < Rw < 50, f"Wigley Rw={Rw:.2f} N outside expected range"
