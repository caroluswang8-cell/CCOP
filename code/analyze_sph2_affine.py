"""Analyze a strict SPH2 star--star affine-drop trajectory against its ODE reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


def affine_rhs(_t: float, y: np.ndarray, radius: float, omega: float) -> np.ndarray:
    """ODE for x semi-axis a and logarithmic rate delta=a_dot/a."""
    a, delta = y
    ratio = radius**4 / a**4
    delta_dot = ((ratio - 1.0) / (ratio + 1.0)) * (delta**2 + omega**2)
    return np.array([delta * a, delta_dot])


def solve_reference(times: np.ndarray, radius: float, delta0: float, omega: float):
    sol = solve_ivp(
        affine_rhs,
        (0.0, float(times[-1])),
        np.array([radius, delta0]),
        args=(radius, omega),
        method="DOP853",
        rtol=2.0e-13,
        atol=2.0e-15,
        dense_output=True,
        max_step=0.02,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    a, delta = sol.sol(times)
    b = radius**2 / a
    ratio = radius**4 / a**4
    delta_dot = ((ratio - 1.0) / (ratio + 1.0)) * (delta**2 + omega**2)
    p_center = 0.5 * a**2 * (delta_dot + delta**2 + omega**2)
    q = np.log(a / radius)
    return sol, a, b, delta, p_center, q


def fit_affine_series(X: np.ndarray, x: np.ndarray, boundary_idx: np.ndarray, radius: float):
    Xc = X - np.mean(X, axis=0)
    design = np.column_stack((Xc, np.ones(len(Xc))))
    ntime = x.shape[0]
    A = np.empty((ntime, 2, 2))
    centroid = np.empty((ntime, 2))
    residual = np.empty(ntime)
    axis_x = np.empty(ntime)
    axis_y = np.empty(ntime)
    boundary_long = np.empty(ntime)
    boundary_short = np.empty(ntime)
    for k in range(ntime):
        coef, *_ = np.linalg.lstsq(design, x[k], rcond=None)
        A[k] = coef[:2].T
        centroid[k] = coef[2]
        pred = design @ coef
        residual[k] = np.sqrt(np.mean(np.sum((x[k] - pred) ** 2, axis=1))) / radius

        # Label-aware lab-frame semi-axes retain the x/y orientation and therefore
        # distinguish a full cycle from the half-cycle of an unordered long axis.
        axis_x[k] = radius * np.linalg.norm(A[k, :, 0])
        axis_y[k] = radius * np.linalg.norm(A[k, :, 1])

        xb = x[k, boundary_idx] - np.mean(x[k, boundary_idx], axis=0)
        cov = xb.T @ xb / len(xb)
        eig = np.linalg.eigvalsh(cov)
        boundary_short[k], boundary_long[k] = np.sqrt(2.0 * np.maximum(eig, 0.0))
    long_axis = np.maximum(axis_x, axis_y)
    short_axis = np.minimum(axis_x, axis_y)
    if np.any(axis_x <= 0.0) or np.any(axis_y <= 0.0):
        raise ValueError("Fitted semi-axes must remain positive before taking logarithms.")
    q = 0.5 * np.log(axis_x / axis_y)
    return {
        "A": A,
        "centroid": centroid,
        "affine_residual": residual,
        "axis_x": axis_x,
        "axis_y": axis_y,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "q": q,
        "boundary_long": boundary_long,
        "boundary_short": boundary_short,
    }


def refine_peak_time(t: np.ndarray, y: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(t) - 1:
        return float(t[i])
    coeff = np.polyfit(t[i - 1 : i + 2], y[i - 1 : i + 2], 2)
    if coeff[0] >= 0.0:
        return float(t[i])
    tp = -coeff[1] / (2.0 * coeff[0])
    if t[i - 1] <= tp <= t[i + 1]:
        return float(tp)
    return float(t[i])


def period_from_signal(t: np.ndarray, y: np.ndarray, expected: float, prominence: float):
    positive_dt = np.diff(t)
    save_dt = float(np.median(positive_dt[positive_dt > 0]))
    distance = max(2, int(0.65 * expected / save_dt))
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
    peak_times = np.array([refine_peak_time(t, y, int(i)) for i in peaks])
    periods = np.diff(peak_times)
    return peak_times, periods


def reference_period(radius: float, delta0: float, omega: float, final_time: float):
    def max_event(t, y, *_args):
        return y[1]

    max_event.direction = -1
    max_event.terminal = False
    event_sol = solve_ivp(
        affine_rhs,
        (0.0, final_time),
        np.array([radius, delta0]),
        args=(radius, omega),
        method="DOP853",
        rtol=2.0e-13,
        atol=2.0e-15,
        events=max_event,
        max_step=0.01,
    )
    peak_times = event_sol.t_events[0]
    return peak_times, np.diff(peak_times)


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def rel_rms(a: np.ndarray, ref: np.ndarray, floor: float = 1.0e-30) -> float:
    return rms(a - ref) / max(rms(ref), floor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}

    t = data["times"].astype(float)
    pressure_t = data["pressure_times"].astype(float)
    X = data["X"].astype(float)
    x = data["x"].astype(float)
    u = data["u"].astype(float)
    boundary_idx = data["free_idx"].astype(int)
    radius = float(data["radius"][0])
    dx = float(data["dx"][0])
    rho = float(data["rho0"][0])
    omega = float(data["omega"][0])
    delta0 = float(data["delta0"][0])
    volume = dx**2

    if not np.all(np.diff(t) > 0):
        raise ValueError("Saved times must be strictly increasing.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        raise ValueError("The trajectory contains non-finite values.")

    ref_sol, a_ref, b_ref, delta_ref, pc_ref, q_ref = solve_reference(
        t, radius, delta0, omega
    )
    _, _, _, _, pc_ref_pressure_t, _ = solve_reference(
        pressure_t, radius, delta0, omega
    )
    fit = fit_affine_series(X, x, boundary_idx, radius)

    long_ref = np.maximum(a_ref, b_ref)
    short_ref = np.minimum(a_ref, b_ref)

    # Exact particle-labelled reference uses the same quadrature as the numerical run.
    stretch = a_ref / radius
    x_ref = np.empty_like(x)
    u_ref = np.empty_like(u)
    x_ref[:, :, 0] = stretch[:, None] * X[None, :, 0]
    x_ref[:, :, 1] = X[None, :, 1] / stretch[:, None]
    u_ref[:, :, 0] = delta_ref[:, None] * x_ref[:, :, 0]
    u_ref[:, :, 1] = -delta_ref[:, None] * x_ref[:, :, 1]

    K_num = 0.5 * rho * volume * np.sum(u**2, axis=(1, 2))
    V_num = 0.5 * rho * omega**2 * volume * np.sum(x**2, axis=(1, 2))
    E_num = K_num + V_num
    K_ref = 0.5 * rho * volume * np.sum(u_ref**2, axis=(1, 2))
    V_ref = 0.5 * rho * omega**2 * volume * np.sum(x_ref**2, axis=(1, 2))
    E_ref = K_ref + V_ref

    represented_area = data["represented_area"].astype(float)
    polygon_area = data["polygon_area"].astype(float)
    exact_area = np.pi * radius**2
    represented_rel_change = represented_area / represented_area[0] - 1.0
    polygon_rel_change = polygon_area / polygon_area[0] - 1.0
    polygon_rel_exact = polygon_area / exact_area - 1.0

    # Determine the exact full signed-shape period first, then use it as a robust
    # minimum-distance scale for numerical peak detection.
    exact_peak_t, exact_periods = reference_period(radius, delta0, omega, float(t[-1]))
    exact_period = float(np.mean(exact_periods))
    peak_t, periods = period_from_signal(
        t, fit["q"], exact_period, prominence=max(0.02, 0.15 * np.ptp(fit["q"]))
    )
    abs_peak_t, abs_periods = period_from_signal(
        t, np.abs(fit["q"]), 0.5 * exact_period, prominence=max(0.01, 0.1 * np.ptp(np.abs(fit["q"])))
    )

    pc_raw = data["center_pressure"].astype(float)
    pc_fit = data["center_pressure_fit"].astype(float)
    valid_pressure = np.isfinite(pc_fit)
    noninitial = valid_pressure & (pressure_t > 0)

    summary = {
        "input": str(args.npz),
        "branch": str(data["branch"][0]),
        "complete_to_requested_final_time": bool(
            abs(t[-1] - float(data["final_time_requested"][0])) < 1.0e-12
        ),
        "n_particles": int(len(X)),
        "n_boundary": int(len(boundary_idx)),
        "n_saved": int(len(t)),
        "saved_dt_median": float(np.median(np.diff(t)[2:])),
        "physical": {
            "radius": radius,
            "dx": dx,
            "R_over_dx": radius / dx,
            "dt": float(data["dt"][0]),
            "T": float(t[-1]),
            "delta0": delta0,
            "omega": omega,
        },
        "period": {
            "exact_signed_shape": exact_period,
            "numerical_signed_shape_mean": float(np.mean(periods)),
            "numerical_signed_shape_std": float(np.std(periods, ddof=1)),
            "relative_error": float(abs(np.mean(periods) - exact_period) / exact_period),
            "n_numerical_cycles": int(len(periods)),
            "numerical_peak_times": peak_t.tolist(),
            "numerical_cycle_periods": periods.tolist(),
            "long_axis_magnitude_recurrence_mean": float(np.mean(abs_periods)),
            "long_axis_magnitude_recurrence_note": "This is approximately half the signed x/y deformation cycle because the major axis swaps orientation.",
        },
        "axes": {
            "long_axis_relative_rms_error": rel_rms(fit["long_axis"], long_ref),
            "short_axis_relative_rms_error": rel_rms(fit["short_axis"], short_ref),
            "long_axis_max_abs_error": float(np.max(np.abs(fit["long_axis"] - long_ref))),
            "short_axis_max_abs_error": float(np.max(np.abs(fit["short_axis"] - short_ref))),
            "boundary_vs_affine_long_rms": rms(fit["boundary_long"] - fit["long_axis"]),
            "boundary_vs_affine_short_rms": rms(fit["boundary_short"] - fit["short_axis"]),
            "max_affine_residual_over_R": float(np.max(fit["affine_residual"])),
            "final_affine_residual_over_R": float(fit["affine_residual"][-1]),
            "max_centroid_displacement": float(np.max(np.linalg.norm(fit["centroid"], axis=1))),
            "numerical_long_range": [float(np.min(fit["long_axis"])), float(np.max(fit["long_axis"]))],
            "exact_long_range": [float(np.min(long_ref)), float(np.max(long_ref))],
            "numerical_short_range": [float(np.min(fit["short_axis"])), float(np.max(fit["short_axis"]))],
            "exact_short_range": [float(np.min(short_ref)), float(np.max(short_ref))],
        },
        "energy": {
            "initial_numeric": float(E_num[0]),
            "initial_exact_same_quadrature": float(E_ref[0]),
            "max_abs_relative_drift": float(np.max(np.abs(E_num / E_num[0] - 1.0))),
            "final_relative_drift": float(E_num[-1] / E_num[0] - 1.0),
            "relative_rms_error_vs_exact": rel_rms(E_num, E_ref),
            "kinetic_relative_rms_error": rel_rms(K_num, K_ref),
            "potential_relative_rms_error": rel_rms(V_num, V_ref),
            "kinetic_error_rms_normalized_by_E0": rms(K_num - K_ref) / E_ref[0],
            "potential_error_rms_normalized_by_E0": rms(V_num - V_ref) / E_ref[0],
        },
        "area": {
            "exact": exact_area,
            "represented_initial": float(represented_area[0]),
            "polygon_initial": float(polygon_area[0]),
            "represented_initial_quadrature_bias_vs_exact": float(represented_area[0] / exact_area - 1.0),
            "polygon_initial_bias_vs_exact": float(polygon_area[0] / exact_area - 1.0),
            "represented_max_abs_relative_change": float(np.max(np.abs(represented_rel_change))),
            "represented_final_relative_change": float(represented_rel_change[-1]),
            "polygon_max_abs_relative_change": float(np.max(np.abs(polygon_rel_change))),
            "polygon_final_relative_change": float(polygon_rel_change[-1]),
            "polygon_max_abs_relative_error_vs_exact": float(np.max(np.abs(polygon_rel_exact))),
            "polygon_final_relative_error_vs_exact": float(polygon_rel_exact[-1]),
        },
        "center_pressure": {
            "fit_relative_rms_error": rel_rms(pc_fit[noninitial], pc_ref_pressure_t[noninitial]),
            "fit_max_abs_error": float(np.max(np.abs(pc_fit[noninitial] - pc_ref_pressure_t[noninitial]))),
            "fit_final_error": float(pc_fit[-1] - pc_ref_pressure_t[-1]),
            "raw_relative_rms_error": rel_rms(pc_raw[noninitial], pc_ref_pressure_t[noninitial]),
            "raw_max_abs_error": float(np.max(np.abs(pc_raw[noninitial] - pc_ref_pressure_t[noninitial]))),
            "archive_reference_max_disagreement": float(np.max(np.abs(data["center_pressure_ref"] - pc_ref_pressure_t))),
            "numerical_fit_range_excluding_initial_placeholder": [float(np.min(pc_fit[noninitial])), float(np.max(pc_fit[noninitial]))],
            "exact_range": [float(np.min(pc_ref_pressure_t)), float(np.max(pc_ref_pressure_t))],
        },
        "strictness": {
            "max_anchor_divergence_rms_saved": float(np.nanmax(data["rms_D_anchor_after"])),
            "max_terminal_divergence_rms_saved": float(np.nanmax(data["rms_D_terminal"])),
            "max_terminal_divergence_max_saved": float(np.nanmax(data["max_D_terminal"])),
            "max_hard_residual": float(np.nanmax(data["hard_resid_rel"])),
            "max_geometry_residual": float(np.nanmax(data["geometry_resid_rel_hard"])),
            "min_raw_terminal_J": float(np.nanmin(data["min_J_terminal"])),
            "max_raw_terminal_J": float(np.nanmax(data["max_J_terminal"])),
        },
    }

    (args.out / "sph2_star_star_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_columns = {
        "time": t,
        "long_axis": fit["long_axis"],
        "short_axis": fit["short_axis"],
        "long_axis_exact": long_ref,
        "short_axis_exact": short_ref,
        "signed_shape_q": fit["q"],
        "signed_shape_q_exact": q_ref,
        "affine_residual_over_R": fit["affine_residual"],
        "K": K_num,
        "V": V_num,
        "E": E_num,
        "K_exact": K_ref,
        "V_exact": V_ref,
        "E_exact": E_ref,
        "represented_area": represented_area,
        "polygon_area": polygon_area,
        "center_pressure_fit": pc_fit,
        "center_pressure_raw": pc_raw,
        "center_pressure_exact": pc_ref_pressure_t,
    }
    with (args.out / "sph2_star_star_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns.keys())
        writer.writerows(zip(*csv_columns.values()))

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    blue, orange, green, gray, red = "#0072B2", "#D55E00", "#009E73", "#4D4D4D", "#CC3311"
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.6), constrained_layout=True)
    legend_kw = dict(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    ax = axes[0, 0]
    ax.plot(t, long_ref, color=gray, ls="--", label="Exact major")
    ax.plot(t, short_ref, color=gray, ls=":", label="Exact minor")
    ax.plot(t, fit["long_axis"], color=blue, label="SPH2 major")
    ax.plot(t, fit["short_axis"], color=orange, label="SPH2 minor")
    ax.set(ylabel="semi-axis", title="a  Major and minor semi-axes")
    ax.legend(ncol=2, **legend_kw)

    ax = axes[0, 1]
    ax.plot(t, q_ref, color=gray, ls="--", label="Exact")
    ax.plot(t, fit["q"], color=blue, label="SPH2")
    ax.scatter(peak_t, np.interp(peak_t, t, fit["q"]), s=10, color=red, zorder=3, label="Cycle peaks")
    ax.set(ylabel="signed deformation q", title="b  Signed deformation and full-cycle timing")
    ax.legend(ncol=3, **legend_kw)

    ax = axes[1, 0]
    ax.plot(t, K_num, color=blue, label=r"$K_h$")
    ax.plot(t, V_num, color=orange, label=r"$V_h$")
    ax.plot(t, E_num, color=green, label=r"$E_h$")
    ax.plot(t, E_ref, color=gray, ls="--", label=r"$E_{\rm ex}$")
    ax.set(ylabel="energy", title="c  Kinetic, potential, and total energy")
    ax.legend(ncol=4, **legend_kw)

    ax = axes[1, 1]
    ax.plot(t, E_num / E_num[0] - 1.0, color=green, label="Total drift")
    ax.plot(t, (K_num - K_ref) / E_ref[0], color=blue, label="Kinetic error")
    ax.plot(t, (V_num - V_ref) / E_ref[0], color=orange, label="Potential error")
    ax.axhline(0.0, color=gray, lw=0.7)
    ax.set(ylabel="normalized error", title="d  Energy error against affine reference")
    ax.legend(ncol=3, **legend_kw)

    ax = axes[2, 0]
    ax.plot(t, represented_rel_change, color=blue, label=r"Represented $(A_J/A_J^0-1)$")
    ax.plot(t, polygon_rel_change, color=orange, label=r"Polygon $(A_\Gamma/A_\Gamma^0-1)$")
    ax.axhline(0.0, color=gray, lw=0.7)
    ax.set(xlabel="time", ylabel="relative area change", title="e  Area preservation")
    ax.legend(ncol=2, **legend_kw)

    ax = axes[2, 1]
    ax.plot(pressure_t, pc_ref_pressure_t, color=gray, ls="--", label="Exact")
    ax.plot(pressure_t[noninitial], pc_fit[noninitial], color=blue, label="SPH2 quadratic fit")
    ax.plot(pressure_t[noninitial], pc_raw[noninitial], color=orange, alpha=0.55, label="SPH2 center particle")
    ax.set(xlabel="time", ylabel=r"$p_c$", title="f  Center pressure")
    ax.legend(ncol=3, **legend_kw)

    for ax in axes.flat:
        ax.set_title(ax.get_title(), pad=42)
        ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.7)
        ax.tick_params(direction="out", length=3, width=0.6)
    fig.savefig(args.out / "sph2_star_star_affine_comparison.pdf", bbox_inches="tight")
    fig.savefig(args.out / "sph2_star_star_affine_comparison.svg", bbox_inches="tight")
    fig.savefig(args.out / "sph2_star_star_affine_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.out / "sph2_star_star_affine_comparison.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
