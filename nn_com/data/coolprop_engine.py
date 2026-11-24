"""Exact PyTorch CoolProp LUT evaluator and supervisory mode controller."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def pressure_guard_torch(x: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch port of validate_com_fig2.m pressure_guard (out-of-place for autograd)."""
    p_in = torch.clamp(x[..., 5], min=0.5e5, max=20.0e5)
    p_out = torch.clamp(x[..., 6], min=0.5e5, max=50.0e5)
    mask = p_out < 1.5 * p_in
    p_in_g = torch.where(mask, torch.clamp(p_in, min=2.0e5), p_in)
    p_out_g = torch.where(mask, torch.maximum(p_out, torch.clamp(p_in_g * 1.5, min=6.0e5)), p_out)
    return torch.stack([
        x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4],
        p_in_g, p_out_g, x[..., 7], x[..., 8],
    ], dim=-1)


def pressure_guard_np(x: np.ndarray) -> np.ndarray:
    """NumPy version of pressure_guard matching validate_com_fig2.m."""
    xp = np.array(x, dtype=np.float64, copy=True)
    p_in = np.clip(xp[..., 5], 0.5e5, 20.0e5)
    p_out = np.clip(xp[..., 6], 0.5e5, 50.0e5)
    mask = p_out < 1.5 * p_in
    p_in = np.where(mask, np.maximum(p_in, 2.0e5), p_in)
    p_out = np.where(mask, np.maximum(p_out, np.maximum(p_in * 1.5, 6.0e5)), p_out)
    xp[..., 5] = p_in
    xp[..., 6] = p_out
    return xp


class CoolPropLUTTorch(nn.Module):
    """Vectorized, differentiable-safe PyTorch lookup matching Source Files/estimateParameters.m."""

    def __init__(self, lut_path: Path):
        super().__init__()
        z = np.load(lut_path)
        for key in z.files:
            val = torch.tensor(z[key], dtype=torch.float64)
            self.register_buffer(key, val)

    def _interp1(self, grid: torch.Tensor, values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_c = torch.clamp(x, min=grid[0], max=grid[-1])
        idx = torch.searchsorted(grid, x_c, right=False) - 1
        idx = torch.clamp(idx, min=0, max=grid.numel() - 2)
        x0 = grid[idx]
        x1 = grid[idx + 1]
        w = (x_c - x0) / (x1 - x0 + 1e-18)
        return (1.0 - w) * values[idx] + w * values[idx + 1]

    def _interp2(self, gx: torch.Tensor, gy: torch.Tensor, table: torch.Tensor,
                 x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xc = torch.clamp(x, min=gx[0], max=gx[-1])
        yc = torch.clamp(y, min=gy[0], max=gy[-1])
        ix = torch.clamp(torch.searchsorted(gx, xc, right=False) - 1, min=0, max=gx.numel() - 2)
        iy = torch.clamp(torch.searchsorted(gy, yc, right=False) - 1, min=0, max=gy.numel() - 2)
        wx = (xc - gx[ix]) / (gx[ix + 1] - gx[ix] + 1e-18)
        wy = (yc - gy[iy]) / (gy[iy + 1] - gy[iy] + 1e-18)
        v00 = table[ix, iy]
        v10 = table[ix + 1, iy]
        v01 = table[ix, iy + 1]
        v11 = table[ix + 1, iy + 1]
        return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11

    @torch.no_grad()
    def evaluate(
        self,
        x_raw: torch.Tensor,
        env_t_k: torch.Tensor,
        speed_kmh: torch.Tensor,
        bat_i: torch.Tensor,
        mot_q: torch.Tensor,
        inv_q: torch.Tensor,
        dcdc_q: torch.Tensor,
        fan_rpm_prev: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate all thermophysical properties at guarded state x_raw [B, 9]."""
        xp = pressure_guard_torch(x_raw)
        p_in = xp[:, 5]
        p_out = xp[:, 6]
        cab_air_t = xp[:, 8]

        p_grid = self.p_grid
        t_air_grid = self.t_air_grid

        comp_v_nom = self.comp_v_nominal_in.squeeze().expand_as(p_in)
        ref_tsat_low = self._interp1(p_grid, self.tsat_vap, p_in)
        comp_hin = self._interp1(p_grid, self.comp_hin, p_in)
        comp_houts = self._interp2(self.p_in_grid, self.p_out_grid, self.h2s_grid, p_in, p_out)

        ref_tsat_high = self._interp1(p_grid, self.tsat_vap, p_out)
        ref_cp_vap_high = self._interp1(p_grid, self.cp_vap, p_out)
        ref_tsat_liq_high = self._interp1(p_grid, self.tsat_liq, p_out)
        ref_hsat_liq_high = self._interp1(p_grid, self.hsat_liq, p_out)
        ref_cp_liq_high = self._interp1(p_grid, self.cp_liq, p_out)
        ref_tsat_liq_low = self._interp1(p_grid, self.tsat_liq, p_in)

        cab_air_cp = self._interp1(t_air_grid, self.air_cp, cab_air_t)
        cab_air_k = self._interp1(t_air_grid, self.air_k, cab_air_t)
        cab_air_mu = self._interp1(t_air_grid, self.air_mu, cab_air_t)
        cab_air_rho = self._interp1(t_air_grid, self.air_rho, cab_air_t)

        # Fix hardcoded 263.15 K from estimateParameters.m -> use dynamic env_t_k
        env_air_cp = self._interp1(t_air_grid, self.air_cp, env_t_k)
        env_air_k = self._interp1(t_air_grid, self.air_k, env_t_k)
        env_air_mu = self._interp1(t_air_grid, self.air_mu, env_t_k)
        env_air_rho = self._interp1(t_air_grid, self.air_rho, env_t_k)

        inner_ref_rho = self._interp1(p_grid, self.rho_vap, p_out)
        inner_ref_k = self._interp1(p_grid, self.k_vap, p_out)
        inner_ref_mu = self._interp1(p_grid, self.mu_vap, p_out)
        inner_ref_cp = ref_cp_vap_high

        evap_ref_rho = self._interp1(p_grid, self.rho_liq, p_in)
        evap_ref_k = self._interp1(p_grid, self.k_liq, p_in)
        evap_ref_mu = self._interp1(p_grid, self.mu_liq, p_in)
        evap_ref_cp = self._interp1(p_grid, self.cp_liq, p_in)

        d_tesat_dp1 = self._interp1(p_grid, self.d_tsat_dp, p_in)
        d_rhoh_l_dp1 = self._interp1(p_grid, self.d_rhoh_l_dp, p_in)
        d_rhoh_g_dp1 = self._interp1(p_grid, self.d_rhoh_g_dp, p_in)

        d_tcsat_dp2 = self._interp1(p_grid, self.d_tsat_dp, p_out)
        d_rhoh_l_dp2 = self._interp1(p_grid, self.d_rhoh_l_dp, p_out)
        d_rhoh_g_dp2 = self._interp1(p_grid, self.d_rhoh_g_dp, p_out)

        ones = torch.ones_like(p_in)
        return {
            "envT": env_t_k,
            "clntRho": 1113.0 * ones,
            "clntK": 0.254 * ones,
            "clntMu": 16.1e-3 * ones,
            "clntCp": 2430.0 * ones,
            "cabAirCp": cab_air_cp,
            "cabAirK": cab_air_k,
            "cabAirMu": cab_air_mu,
            "cabAirRho": cab_air_rho,
            "innerCondRefRho": inner_ref_rho,
            "innerCondRefK": inner_ref_k,
            "innerCondRefMu": inner_ref_mu,
            "innerCondRefCp": inner_ref_cp,
            "condEvapRefRho": evap_ref_rho,
            "condEvapRefK": evap_ref_k,
            "condEvapRefMu": evap_ref_mu,
            "condEvapRefCp": evap_ref_cp,
            "condEvapAirRho": env_air_rho,
            "condEvapAirK": env_air_k,
            "condEvapAirMu": env_air_mu,
            "condEvapAirCp": env_air_cp,
            "evapRefRho": evap_ref_rho,
            "evapRefK": evap_ref_k,
            "evapRefMu": evap_ref_mu,
            "evapRefCp": evap_ref_cp,
            "chilRefRho": evap_ref_rho,
            "chilRefK": evap_ref_k,
            "chilRefMu": evap_ref_mu,
            "chilRefCp": evap_ref_cp,
            "speed": speed_kmh,
            "compHin": comp_hin,
            "compHoutS": comp_houts,
            "compVnominalIn": comp_v_nom,
            "refTsatHighPK": ref_tsat_high,
            "refCpVapHighP": ref_cp_vap_high,
            "refHsatLiqHighP": ref_hsat_liq_high,
            "refTsatLiqHighPK": ref_tsat_liq_high,
            "refTsatLiqLowPK": ref_tsat_liq_low,
            "refCpLiqHighP": ref_cp_liq_high,
            "motQmeas": mot_q,
            "invQmeas": inv_q,
            "dcdcQmeas": dcdc_q,
            "batImeas": bat_i,
            "evapVoidFrac": 0.8 * ones,
            "condVoidFrac": 0.5 * ones,
            "evapVol_m3": 0.001 * ones,
            "condVol_m3": 0.001 * ones,
            "evapWallThermalMass_J_K": 5000.0 * ones,
            "condWallThermalMass_J_K": 5000.0 * ones,
            "dTesat_dp1": d_tesat_dp1,
            "d_rhoh_l_dp1": d_rhoh_l_dp1,
            "d_rhoh_g_dp1": d_rhoh_g_dp1,
            "dTcsat_dp2": d_tcsat_dp2,
            "d_rhoh_l_dp2": d_rhoh_l_dp2,
            "d_rhoh_g_dp2": d_rhoh_g_dp2,
            "fanRpm_prev": fan_rpm_prev,
        }


def compute_supervisory_sequence(
    tamb_c: float,
    clnt_inv_tout_k: np.ndarray,
    clnt_bat_tin_k: np.ndarray,
    comp_n: np.ndarray,
    pump_n: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Exact Python port of Source Files/thermalLogicAndEstimation.m over a trajectory."""
    n = len(comp_n)
    env_t_k = tamb_c + 273.15

    thermal_mode_arr = np.zeros(n, dtype=np.float64)
    cmd_blend_air = np.zeros(n, dtype=np.float64)
    cmd_exv_hp = np.zeros(n, dtype=np.float64)
    cmd_exv_evap = np.zeros(n, dtype=np.float64)
    cmd_exv_chiller = np.zeros(n, dtype=np.float64)
    cmd_ps = np.ones(n, dtype=np.float64)
    cmd_heater = np.zeros(n, dtype=np.float64)

    prev_mode = 0
    clnt_bat_filt = float(clnt_bat_tin_k[0])
    cmd_exv_chiller_set = 0
    heater_flag = 0

    if env_t_k >= 273.15 + 21.0:
        setpoint = 1
    elif env_t_k < 273.15 + 19.0:
        setpoint = 2
    else:
        setpoint = 0

    for k in range(n):
        clnt_bat_filt = 0.8 * clnt_bat_filt + 0.2 * float(clnt_bat_tin_k[k])
        cn = float(comp_n[k])
        pn = float(pump_n[k])
        c_inv = float(clnt_inv_tout_k[k])
        c_bat = float(clnt_bat_tin_k[k])

        if prev_mode != 0 and setpoint == 0:
            mode = 0
        elif prev_mode == 0 and setpoint != 0 and cn >= 10.0:
            mode = setpoint
        else:
            mode = prev_mode

        if clnt_bat_filt >= 273.15 + 35.0:
            cmd_exv_chiller_set = 1
        elif clnt_bat_filt <= 273.15 + 25.0:
            cmd_exv_chiller_set = 0

        ps = 1.0
        if mode <= 1:
            t_cmp = max(env_t_k, c_inv)
            if t_cmp >= 273.15 + 35.0:
                ps = 0.0
            elif t_cmp <= 273.15 + 30.0:
                ps = 1.0

        if heater_flag == 0:
            if env_t_k < (273.15 + 5.0) and c_bat < (273.15 + 15.0) and pn > 0.0:
                heater_flag = 1
        else:
            if env_t_k > (273.15 + 5.0) or c_bat > (273.15 + 5.0) or pn <= 0.0:
                heater_flag = 0

        thermal_mode_arr[k] = float(mode)
        cmd_blend_air[k] = 1.0 if mode == 2 else 0.0
        cmd_exv_hp[k] = 1.0 if mode == 2 else 0.0
        cmd_exv_evap[k] = 1.0 if mode == 1 else 0.0
        cmd_exv_chiller[k] = (cmd_exv_chiller_set * 0.024) if mode == 1 else 0.0
        cmd_ps[k] = ps
        cmd_heater[k] = float(heater_flag)
        prev_mode = mode

    return {
        "thermalMode": thermal_mode_arr,
        "cmdBlendAir": cmd_blend_air,
        "cmdExvHeatpump": cmd_exv_hp,
        "cmdExvEvap": cmd_exv_evap,
        "cmdExvChiller": cmd_exv_chiller,
        "cmdParallelSerial": cmd_ps,
        "cmdHeater": cmd_heater,
    }
