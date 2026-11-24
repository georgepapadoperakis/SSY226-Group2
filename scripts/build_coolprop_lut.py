"""Build exact CoolProp R134a & Dry Air lookup tables for vectorized PyTorch COM."""

from pathlib import Path
import numpy as np
from CoolProp.CoolProp import PropsSI


def safe_props(out_key: str, k1: str, v1: float, k2: str, v2: float, fluid: str, p_crit: float) -> float:
    """Evaluate CoolProp PropsSI with safe saturation clamping near/above P_crit."""
    try:
        return float(PropsSI(out_key, k1, float(v1), k2, float(v2), fluid))
    except ValueError:
        if k1 == "P" and k2 == "Q":
            p_clamped = min(float(v1), 0.995 * p_crit)
            return float(PropsSI(out_key, k1, p_clamped, k2, float(v2), fluid))
        if k1 == "P" and k2 == "S":
            return float(PropsSI(out_key, k1, float(v1) + 100.0, k2, float(v2), fluid))
        raise


def build_lut(out_path: Path) -> None:
    fluid_ref = "R134a"
    fluid_air = "Air"
    p_atm = 101325.0
    p_crit = float(PropsSI("Pcrit", fluid_ref))

    # Nominal compressor inlet specific volume (3 bar, 11 C)
    rho_nom_in = float(PropsSI("D", "P", 3.0e5, "T", 11.0 + 273.15, fluid_ref))
    comp_v_nominal_in = 1.0 / rho_nom_in

    # 1D R134a saturation grid (log-spaced from 0.4 bar to 55 bar; sat capped just below P_crit)
    n_p = 2000
    p_grid = np.geomspace(0.4e5, 55.0e5, n_p)
    p_sat_grid = np.minimum(p_grid, 0.995 * p_crit)

    tsat_vap = np.empty(n_p, dtype=np.float64)
    tsat_liq = np.empty(n_p, dtype=np.float64)
    hsat_liq = np.empty(n_p, dtype=np.float64)
    hsat_vap = np.empty(n_p, dtype=np.float64)
    cp_liq = np.empty(n_p, dtype=np.float64)
    cp_vap = np.empty(n_p, dtype=np.float64)
    rho_liq = np.empty(n_p, dtype=np.float64)
    rho_vap = np.empty(n_p, dtype=np.float64)
    k_liq = np.empty(n_p, dtype=np.float64)
    k_vap = np.empty(n_p, dtype=np.float64)
    mu_liq = np.empty(n_p, dtype=np.float64)
    mu_vap = np.empty(n_p, dtype=np.float64)
    d_tsat_dp = np.empty(n_p, dtype=np.float64)
    d_rhoh_l_dp = np.empty(n_p, dtype=np.float64)
    d_rhoh_g_dp = np.empty(n_p, dtype=np.float64)
    comp_hin = np.empty(n_p, dtype=np.float64)
    comp_sin = np.empty(n_p, dtype=np.float64)

    dp = 100.0
    superheat_k = 2.0

    for i, p_sat in enumerate(p_sat_grid):
        tsat_vap[i] = safe_props("T", "P", p_sat, "Q", 1, fluid_ref, p_crit)
        tsat_liq[i] = safe_props("T", "P", p_sat, "Q", 0, fluid_ref, p_crit)
        hsat_liq[i] = safe_props("H", "P", p_sat, "Q", 0, fluid_ref, p_crit)
        hsat_vap[i] = safe_props("H", "P", p_sat, "Q", 1, fluid_ref, p_crit)
        cp_liq[i] = safe_props("C", "P", p_sat, "Q", 0, fluid_ref, p_crit)
        cp_vap[i] = safe_props("C", "P", p_sat, "Q", 1, fluid_ref, p_crit)
        rho_liq[i] = safe_props("D", "P", p_sat, "Q", 0, fluid_ref, p_crit)
        rho_vap[i] = safe_props("D", "P", p_sat, "Q", 1, fluid_ref, p_crit)
        k_liq[i] = safe_props("L", "P", p_sat, "Q", 0, fluid_ref, p_crit)
        k_vap[i] = safe_props("L", "P", p_sat, "Q", 1, fluid_ref, p_crit)
        mu_liq[i] = safe_props("V", "P", p_sat, "Q", 0, fluid_ref, p_crit)
        mu_vap[i] = safe_props("V", "P", p_sat, "Q", 1, fluid_ref, p_crit)

        p_plus = min(p_sat + dp, 0.996 * p_crit)
        p_minus = max(p_sat - dp, 0.35e5)
        span = p_plus - p_minus

        t_p = safe_props("T", "P", p_plus, "Q", 1, fluid_ref, p_crit)
        t_m = safe_props("T", "P", p_minus, "Q", 1, fluid_ref, p_crit)
        d_tsat_dp[i] = (t_p - t_m) / span

        rl_p = safe_props("D", "P", p_plus, "Q", 0, fluid_ref, p_crit)
        hl_p = safe_props("H", "P", p_plus, "Q", 0, fluid_ref, p_crit)
        rl_m = safe_props("D", "P", p_minus, "Q", 0, fluid_ref, p_crit)
        hl_m = safe_props("H", "P", p_minus, "Q", 0, fluid_ref, p_crit)
        d_rhoh_l_dp[i] = (rl_p * hl_p - rl_m * hl_m) / span

        rg_p = safe_props("D", "P", p_plus, "Q", 1, fluid_ref, p_crit)
        hg_p = safe_props("H", "P", p_plus, "Q", 1, fluid_ref, p_crit)
        rg_m = safe_props("D", "P", p_minus, "Q", 1, fluid_ref, p_crit)
        hg_m = safe_props("H", "P", p_minus, "Q", 1, fluid_ref, p_crit)
        d_rhoh_g_dp[i] = (rg_p * hg_p - rg_m * hg_m) / span

        t_in = tsat_vap[i] + superheat_k
        h1 = float(PropsSI("H", "P", float(p_sat), "T", t_in, fluid_ref))
        s1 = float(PropsSI("S", "P", float(p_sat), "H", h1, fluid_ref))
        comp_hin[i] = h1
        comp_sin[i] = s1

    # 2D isentropic outlet enthalpy grid h2s(p_in, p_out)
    p_in_grid = np.geomspace(0.4e5, 20.0e5, 120)
    p_out_grid = np.geomspace(0.5e5, 55.0e5, 150)
    h2s_grid = np.empty((len(p_in_grid), len(p_out_grid)), dtype=np.float64)

    for i, pin in enumerate(p_in_grid):
        p_in_sat = min(pin, 0.995 * p_crit)
        tsat_in = safe_props("T", "P", p_in_sat, "Q", 1, fluid_ref, p_crit)
        h1 = float(PropsSI("H", "P", float(pin), "T", tsat_in + superheat_k, fluid_ref))
        s1 = float(PropsSI("S", "P", float(pin), "H", h1, fluid_ref))
        for j, pout in enumerate(p_out_grid):
            h2s_grid[i, j] = safe_props("H", "P", float(pout), "S", s1, fluid_ref, p_crit)

    # 1D Dry Air grid
    t_air_grid = np.linspace(220.0, 350.0, 500)
    air_cp = np.array([float(PropsSI("C", "T", float(t), "P", p_atm, fluid_air)) for t in t_air_grid])
    air_k = np.array([float(PropsSI("L", "T", float(t), "P", p_atm, fluid_air)) for t in t_air_grid])
    air_mu = np.array([float(PropsSI("V", "T", float(t), "P", p_atm, fluid_air)) for t in t_air_grid])
    air_rho = np.array([float(PropsSI("D", "T", float(t), "P", p_atm, fluid_air)) for t in t_air_grid])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        p_crit=np.array([p_crit]),
        comp_v_nominal_in=np.array([comp_v_nominal_in]),
        p_grid=p_grid,
        tsat_vap=tsat_vap,
        tsat_liq=tsat_liq,
        hsat_liq=hsat_liq,
        hsat_vap=hsat_vap,
        cp_liq=cp_liq,
        cp_vap=cp_vap,
        rho_liq=rho_liq,
        rho_vap=rho_vap,
        k_liq=k_liq,
        k_vap=k_vap,
        mu_liq=mu_liq,
        mu_vap=mu_vap,
        d_tsat_dp=d_tsat_dp,
        d_rhoh_l_dp=d_rhoh_l_dp,
        d_rhoh_g_dp=d_rhoh_g_dp,
        comp_hin=comp_hin,
        comp_sin=comp_sin,
        p_in_grid=p_in_grid,
        p_out_grid=p_out_grid,
        h2s_grid=h2s_grid,
        t_air_grid=t_air_grid,
        air_cp=air_cp,
        air_k=air_k,
        air_mu=air_mu,
        air_rho=air_rho,
    )
    print(f"Saved CoolProp LUT to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Verify against 200 random test points within subcritical operating range
    rng = np.random.default_rng(123)
    p1_test = rng.uniform(0.6e5, 4.0e5, 200)
    p2_test = rng.uniform(8.0e5, 39.0e5, 200)
    t_test = rng.uniform(240.0, 310.0, 200)

    def interp1(grid, vals, x):
        return np.interp(x, grid, vals)

    def interp2(gx, gy, z, x, y):
        ix = np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2)
        iy = np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2)
        wx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
        wy = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
        return (
            (1 - wx) * (1 - wy) * z[ix, iy]
            + wx * (1 - wy) * z[ix + 1, iy]
            + (1 - wx) * wy * z[ix, iy + 1]
            + wx * wy * z[ix + 1, iy + 1]
        )

    errs = {k: 0.0 for k in ["tsat_low", "compHin", "compHoutS", "d_rhoh_l_dp1", "d_rhoh_g_dp2", "air_cp", "air_rho"]}
    h2s_pred = interp2(p_in_grid, p_out_grid, h2s_grid, p1_test, p2_test)
    for idx in range(200):
        p1, p2, tc = float(p1_test[idx]), float(p2_test[idx]), float(t_test[idx])
        ts1 = float(PropsSI("T", "P", p1, "Q", 1, fluid_ref))
        h1 = float(PropsSI("H", "P", p1, "T", ts1 + superheat_k, fluid_ref))
        s1 = float(PropsSI("S", "P", p1, "H", h1, fluid_ref))
        h2s = float(PropsSI("H", "P", p2, "S", s1, fluid_ref))

        drl_1 = (
            PropsSI("D", "P", p1 + dp, "Q", 0, fluid_ref) * PropsSI("H", "P", p1 + dp, "Q", 0, fluid_ref)
            - PropsSI("D", "P", p1 - dp, "Q", 0, fluid_ref) * PropsSI("H", "P", p1 - dp, "Q", 0, fluid_ref)
        ) / (2 * dp)
        drg_2 = (
            PropsSI("D", "P", p2 + dp, "Q", 1, fluid_ref) * PropsSI("H", "P", p2 + dp, "Q", 1, fluid_ref)
            - PropsSI("D", "P", p2 - dp, "Q", 1, fluid_ref) * PropsSI("H", "P", p2 - dp, "Q", 1, fluid_ref)
        ) / (2 * dp)
        acp = float(PropsSI("C", "T", tc, "P", p_atm, fluid_air))
        arho = float(PropsSI("D", "T", tc, "P", p_atm, fluid_air))

        errs["tsat_low"] = max(errs["tsat_low"], abs(interp1(p_grid, tsat_vap, p1) - ts1) / abs(ts1))
        errs["compHin"] = max(errs["compHin"], abs(interp1(p_grid, comp_hin, p1) - h1) / abs(h1))
        errs["compHoutS"] = max(errs["compHoutS"], abs(h2s_pred[idx] - h2s) / abs(h2s))
        errs["d_rhoh_l_dp1"] = max(errs["d_rhoh_l_dp1"], abs(interp1(p_grid, d_rhoh_l_dp, p1) - drl_1) / abs(drl_1))
        errs["d_rhoh_g_dp2"] = max(errs["d_rhoh_g_dp2"], abs(interp1(p_grid, d_rhoh_g_dp, p2) - drg_2) / abs(drg_2))
        errs["air_cp"] = max(errs["air_cp"], abs(interp1(t_air_grid, air_cp, tc) - acp) / abs(acp))
        errs["air_rho"] = max(errs["air_rho"], abs(interp1(t_air_grid, air_rho, tc) - arho) / abs(arho))

    print("Max relative errors vs live CoolProp (200 random points):")
    for k, v in errs.items():
        print(f"  {k:16s}: {v:.3e}")
        assert v < 5e-4, f"LUT error too high for {k}: {v}"
    print("[OK] CoolProp LUT verification passed (< 5e-4 max relative error).")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    build_lut(repo_root / "Data" / "coolprop_r134a_air_lut.npz")
