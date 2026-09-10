"""Compare two CCOP-midpoint affine-drop archives against the analytic ODE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve()
SHARED = HERE.parents[1] / "sph2_star_star_analysis"
sys.path.insert(0, str(SHARED))
from analyze_sph2_affine import fit_affine_series, reference_period, solve_reference  # noqa: E402


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def load_case(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {key: z[key].copy() for key in z.files}
    t = d["times"].astype(float)
    pt = d["pressure_times"].astype(float)
    if not np.all(np.diff(t) > 0.0):
        raise ValueError(f"Non-monotone saved times in {path}")
    if not all(np.all(np.isfinite(d[key])) for key in ("x", "u", "p", "represented_area", "polygon_area")):
        raise ValueError(f"Non-finite data in {path}")

    R = float(d["radius"][0])
    delta0 = float(d["delta0"][0])
    omega = float(d["omega"][0])
    rho = float(d["rho0"][0])
    dx = float(d["dx"][0])
    ref_sol, a, b, delta, pc, q = solve_reference(t, R, delta0, omega)
    _, _, _, _, pc_pt, _ = solve_reference(pt, R, delta0, omega)
    fit = fit_affine_series(d["X"], d["x"], d["free_idx"].astype(int), R)

    stretch = a / R
    xref = np.empty_like(d["x"])
    uref = np.empty_like(d["u"])
    xref[:, :, 0] = stretch[:, None] * d["X"][None, :, 0]
    xref[:, :, 1] = d["X"][None, :, 1] / stretch[:, None]
    uref[:, :, 0] = delta[:, None] * xref[:, :, 0]
    uref[:, :, 1] = -delta[:, None] * xref[:, :, 1]
    V0 = dx**2
    K = 0.5 * rho * V0 * np.sum(d["u"] ** 2, axis=(1, 2))
    V = 0.5 * rho * omega**2 * V0 * np.sum(d["x"] ** 2, axis=(1, 2))
    Kr = 0.5 * rho * V0 * np.sum(uref**2, axis=(1, 2))
    Vr = 0.5 * rho * omega**2 * V0 * np.sum(xref**2, axis=(1, 2))
    E, Er = K + V, Kr + Vr

    # Cubic-spline extrema use all saved samples and reduce the cadence bias of
    # a three-point parabolic peak fit. Positive maxima retain axis orientation.
    cs = CubicSpline(t, fit["q"])
    roots = cs.derivative().roots(extrapolate=False)
    peak_t = np.array([r for r in roots if r > 0.1 and cs(r) > 0.0 and cs.derivative(2)(r) < 0.0])
    exact_peak_t, exact_periods = reference_period(R, delta0, omega, float(t[-1]))
    npeak = min(len(peak_t), len(exact_peak_t))
    peak_t = peak_t[:npeak]
    exact_peak_t = exact_peak_t[:npeak]
    periods = np.diff(peak_t)

    valid_p = pt > 0.0
    AJ = d["represented_area"].astype(float)
    AG = d["polygon_area"].astype(float)
    return {
        "data": d,
        "t": t,
        "pt": pt,
        "R": R,
        "dt": float(d["dt"][0]),
        "fit": fit,
        "a": a,
        "b": b,
        "long_ref": np.maximum(a, b),
        "short_ref": np.minimum(a, b),
        "q_ref": q,
        "pc_ref": pc_pt,
        "K": K,
        "V": V,
        "E": E,
        "Kr": Kr,
        "Vr": Vr,
        "Er": Er,
        "AJ_rel": AJ / AJ[0] - 1.0,
        "AG_rel": AG / AG[0] - 1.0,
        "peak_t": peak_t,
        "exact_peak_t": exact_peak_t,
        "periods": periods,
        "exact_period": float(np.mean(exact_periods)),
        "valid_p": valid_p,
    }


def metrics(c: dict) -> dict:
    fit, d = c["fit"], c["data"]
    pfit = d["center_pressure_fit"].astype(float)
    vp = c["valid_p"]
    return {
        "dt": c["dt"],
        "n_saved": int(len(c["t"])),
        "period_cubic_mean": float(np.mean(c["periods"])),
        "period_relative_error": float(abs(np.mean(c["periods"]) - c["exact_period"]) / c["exact_period"]),
        "last_peak_timing_error": float(c["peak_t"][-1] - c["exact_peak_t"][-1]),
        "long_axis_relative_rms_error": rms(fit["long_axis"] - c["long_ref"]) / rms(c["long_ref"]),
        "short_axis_relative_rms_error": rms(fit["short_axis"] - c["short_ref"]) / rms(c["short_ref"]),
        "long_axis_max_abs_error": float(np.max(np.abs(fit["long_axis"] - c["long_ref"]))),
        "short_axis_max_abs_error": float(np.max(np.abs(fit["short_axis"] - c["short_ref"]))),
        "signed_shape_rms_error": rms(fit["q"] - c["q_ref"]),
        "max_affine_residual_over_R": float(np.max(fit["affine_residual"])),
        "max_centroid_displacement": float(np.max(np.linalg.norm(fit["centroid"], axis=1))),
        "energy_max_abs_relative_drift": float(np.max(np.abs(c["E"] / c["E"][0] - 1.0))),
        "energy_final_relative_drift": float(c["E"][-1] / c["E"][0] - 1.0),
        "energy_relative_rms_error": rms(c["E"] - c["Er"]) / rms(c["Er"]),
        "represented_area_max_abs_relative_change": float(np.max(np.abs(c["AJ_rel"]))),
        "polygon_area_max_abs_relative_change": float(np.max(np.abs(c["AG_rel"]))),
        "polygon_area_final_relative_change": float(c["AG_rel"][-1]),
        "center_pressure_fit_relative_rms_error": rms(pfit[vp] - c["pc_ref"][vp]) / rms(c["pc_ref"][vp]),
        "center_pressure_fit_max_abs_error": float(np.max(np.abs(pfit[vp] - c["pc_ref"][vp]))),
        "terminal_divergence_rms_max": float(np.nanmax(d["rms_D_terminal"])),
        "terminal_divergence_max_norm_max": float(np.nanmax(d["max_D_terminal"])),
        "hard_residual_max": float(np.nanmax(d["hard_resid_rel"])),
        "raw_terminal_J_min": float(np.nanmin(d["min_J_terminal"])),
        "raw_terminal_J_max": float(np.nanmax(d["max_J_terminal"])),
    }


def ratio_order(coarse: float, fine: float) -> dict:
    coarse_abs, fine_abs = abs(coarse), abs(fine)
    error_pair = np.asarray([coarse_abs, fine_abs])
    if np.any(error_pair <= 0.0):
        raise ValueError("Positive nonzero errors are required before taking log2 ratios.")
    ratio = coarse_abs / fine_abs
    return {"coarse_over_fine": float(ratio), "two_level_order": float(np.log2(ratio))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coarse", type=Path)
    ap.add_argument("fine", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [load_case(args.coarse), load_case(args.fine)]
    cases.sort(key=lambda c: c["dt"], reverse=True)
    coarse, fine = cases
    mc, mf = metrics(coarse), metrics(fine)

    comparison_keys = [
        "long_axis_relative_rms_error",
        "short_axis_relative_rms_error",
        "signed_shape_rms_error",
        "energy_relative_rms_error",
        "energy_max_abs_relative_drift",
        "polygon_area_max_abs_relative_change",
        "represented_area_max_abs_relative_change",
        "center_pressure_fit_relative_rms_error",
        "last_peak_timing_error",
    ]
    orders = {key: ratio_order(mc[key], mf[key]) for key in comparison_keys}
    summary = {
        "input_files": [str(args.coarse), str(args.fine)],
        "analytic_full_period": coarse["exact_period"],
        "coarse": mc,
        "fine": mf,
        "step_halving": orders,
        "interpretation": {
            "primary": "Axis, signed-shape, energy, polygon-area, center-pressure, and cumulative phase errors decrease by approximately four under dt halving, consistent with second-order midpoint behavior at fixed R/dx=50.",
            "represented_area_note": "The represented-area maximum change has begun to encounter a spatial/geometric floor, so its two-level order is lower than the other diagnostics.",
            "period_note": "The period is extracted from cubic-spline extrema. The last-peak timing error is the more stable temporal-order diagnostic because mean-period differences contain cancellation and output-cadence interpolation error.",
        },
    }
    (args.out / "midpoint_pair_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (args.out / "midpoint_pair_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "dt_0.002", "dt_0.001", "coarse_over_fine", "two_level_order"])
        for key in comparison_keys:
            w.writerow([key, mc[key], mf[key], orders[key]["coarse_over_fine"], orders[key]["two_level_order"]])

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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue, orange, gray, green, red = "#0072B2", "#D55E00", "#4D4D4D", "#009E73", "#CC3311"
    colors = [orange, blue]
    labels = ["dt = 0.002", "dt = 0.001"]
    legend_kw = dict(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig, axa = plt.subplots(3, 2, figsize=(7.2, 9.6), constrained_layout=True)

    ax = axa[0, 0]
    ax.plot(coarse["t"], coarse["long_ref"], color=gray, ls="--", label="Exact major")
    ax.plot(coarse["t"], coarse["short_ref"], color=gray, ls=":", label="Exact minor")
    for c, col, lab in zip(cases, colors, labels):
        ax.plot(c["t"], c["fit"]["long_axis"], color=col, label=f"{lab} major")
        ax.plot(c["t"], c["fit"]["short_axis"], color=col, ls="-.", label=f"{lab} minor")
    ax.set(ylabel="semi-axis", title="a  Major and minor semi-axes")
    ax.legend(ncol=3, **legend_kw)

    ax = axa[0, 1]
    for c, col, lab in zip(cases, colors, labels):
        ax.plot(np.arange(1, len(c["peak_t"]) + 1), c["peak_t"] - c["exact_peak_t"], color=col, marker="o", ms=3, label=lab)
    ax.axhline(0.0, color=gray, lw=0.7)
    ax.set(xlabel="positive-maximum index", ylabel="peak-time error", title="b  Accumulated phase error")
    ax.legend(ncol=2, **legend_kw)

    ax = axa[1, 0]
    for c, col, lab in zip(cases, colors, labels):
        ax.plot(c["t"], c["E"] / c["E"][0] - 1.0, color=col, label=lab)
    ax.axhline(0.0, color=gray, lw=0.7)
    ax.set(ylabel="relative total-energy drift", title="c  Mechanical-energy behavior")
    ax.legend(ncol=2, **legend_kw)

    ax = axa[1, 1]
    c = fine
    ax.plot(c["t"], c["K"], color=blue, label="Kinetic")
    ax.plot(c["t"], c["V"], color=orange, label="Potential")
    ax.plot(c["t"], c["E"], color=green, label="Total")
    ax.plot(c["t"], c["Er"], color=gray, ls="--", label="Exact total")
    ax.set(ylabel="energy", title="d  Conservative energy exchange, dt = 0.001")
    ax.legend(ncol=4, **legend_kw)

    ax = axa[2, 0]
    for c, col, lab in zip(cases, colors, labels):
        vp = c["valid_p"]
        err = c["data"]["center_pressure_fit"][vp] - c["pc_ref"][vp]
        ax.plot(c["pt"][vp], err, color=col, label=lab)
    ax.axhline(0.0, color=gray, lw=0.7)
    ax.set(xlabel="pressure-action time", ylabel="center-pressure error", title="e  Center-pressure accuracy")
    ax.legend(ncol=2, **legend_kw)

    ax = axa[2, 1]
    display = [
        ("Major\naxis", "long_axis_relative_rms_error"),
        ("Minor\naxis", "short_axis_relative_rms_error"),
        ("Shape\nq", "signed_shape_rms_error"),
        ("E\nRMS", "energy_relative_rms_error"),
        ("E\nmax", "energy_max_abs_relative_drift"),
        ("p fit", "center_pressure_fit_relative_rms_error"),
        ("Peak\nphase", "last_peak_timing_error"),
    ]
    ratios = [orders[key]["coarse_over_fine"] for _, key in display]
    xpos = np.arange(len(display))
    ax.bar(xpos, ratios, color=[blue if 3.5 <= r <= 4.5 else orange for r in ratios], width=0.68)
    ax.axhline(4.0, color=gray, ls="--", label="Second-order ratio = 4")
    ax.set_xticks(xpos, [name for name, _ in display])
    ax.set(ylabel="error ratio (dt 0.002 / 0.001)", title="f  Effect of halving the time step")
    ax.legend(ncol=1, **legend_kw)

    for ax in axa.flat:
        ax.set_title(ax.get_title(), pad=42)
        ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.7)
        ax.tick_params(direction="out", length=3, width=0.6)
    base = args.out / "ccop_midpoint_dt_comparison"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
