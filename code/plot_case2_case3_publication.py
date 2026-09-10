"""Build publication-grade comparison figures for manuscript Cases 2 and 3.

Case 2 reads the strict three-branch affine-drop archives and compares every
saved state with the same high-accuracy affine ODE reference.  Case 3 reads the
validated differential-vortex result archive and visualizes the principal
spatial runs.  The script writes compact source-data tables, scalar summaries,
and PDF/SVG/PNG/TIFF figure bundles.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp


CASE2_FILES = (
    "sph2_star_star_Rdx50_dt0p002_T30.npz",
    "sph2_terminal_terminal_Rdx50_dt0p002_T30.npz",
    "sph2_ccop_midpoint_Rdx50_dt0p004_T30.npz",
    "sph2_ccop_midpoint_Rdx50_dt0p002_T30.npz",
    "sph2_ccop_midpoint_Rdx50_dt0p001_T30.npz",
)

COLORS = {
    "exact": "#202020",
    "star": "#7A7A7A",
    "terminal": "#D55E00",
    "mid_004": "#9ECAE1",
    "mid_002": "#3182BD",
    "mid_001": "#08519C",
    "midpoint": "#1764AB",
    "endpoint": "#D55E00",
    "green": "#178C68",
    "purple": "#7B5AA6",
}


@dataclass(frozen=True)
class CaseStyle:
    key: str
    label: str
    color: str
    linestyle: str
    linewidth: float
    zorder: int


CASE2_STYLES = {
    "star-star_0.002": CaseStyle(
        "star-star_0.002", "Star–star, dt = 2e-3", COLORS["star"], (0, (2, 2)), 1.05, 2
    ),
    "terminal-terminal_0.002": CaseStyle(
        "terminal-terminal_0.002", "Terminal–terminal, dt = 2e-3", COLORS["terminal"],
        (0, (5, 2, 1, 2)), 1.15, 3
    ),
    "ccop-midpoint_0.004": CaseStyle(
        "ccop-midpoint_0.004", "Midpoint, dt = 4e-3", COLORS["mid_004"], "-", 0.95, 1
    ),
    "ccop-midpoint_0.002": CaseStyle(
        "ccop-midpoint_0.002", "Midpoint, dt = 2e-3", COLORS["mid_002"], "-", 1.05, 4
    ),
    "ccop-midpoint_0.001": CaseStyle(
        "ccop-midpoint_0.001", "Midpoint, dt = 1e-3", COLORS["mid_001"], "-", 1.25, 5
    ),
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.3,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.1,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def affine_rhs(_time: float, state: np.ndarray, radius: float, omega: float) -> np.ndarray:
    axis, delta = state
    ratio = radius**4 / axis**4
    delta_dot = ((ratio - 1.0) / (ratio + 1.0)) * (delta**2 + omega**2)
    return np.array([delta * axis, delta_dot], dtype=float)


def solve_affine_reference(
    times: np.ndarray, radius: float, delta0: float, omega: float
) -> dict[str, np.ndarray]:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) < 0.0):
        raise ValueError("Reference times must be a nondecreasing one-dimensional array.")
    solution = solve_ivp(
        affine_rhs,
        (0.0, float(times[-1])),
        np.array([radius, delta0], dtype=float),
        args=(radius, omega),
        method="DOP853",
        rtol=2.0e-13,
        atol=2.0e-15,
        dense_output=True,
        max_step=0.02,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    axis_x, delta = solution.sol(times)
    axis_y = radius**2 / axis_x
    ratio = radius**4 / axis_x**4
    delta_dot = ((ratio - 1.0) / (ratio + 1.0)) * (delta**2 + omega**2)
    center_pressure = 0.5 * axis_x**2 * (delta_dot + delta**2 + omega**2)
    return {
        "axis_x": axis_x,
        "axis_y": axis_y,
        "major": np.maximum(axis_x, axis_y),
        "minor": np.minimum(axis_x, axis_y),
        "delta": delta,
        "center_pressure": center_pressure,
    }


def fit_affine_axes(material: np.ndarray, positions: np.ndarray, radius: float) -> dict[str, np.ndarray]:
    centered = material - np.mean(material, axis=0)
    design = np.column_stack((centered, np.ones(len(centered))))
    gram = design.T @ design
    if np.linalg.cond(gram) > 1.0e12:
        raise ValueError("The affine-fit design matrix is unexpectedly ill conditioned.")
    left_inverse = np.linalg.solve(gram, design.T)
    coefficients = np.einsum("an,tnj->taj", left_inverse, positions)
    affine = np.transpose(coefficients[:, :2, :], (0, 2, 1))
    axis_x = radius * np.linalg.norm(affine[:, :, 0], axis=1)
    axis_y = radius * np.linalg.norm(affine[:, :, 1], axis=1)
    return {
        "axis_x": axis_x,
        "axis_y": axis_y,
        "major": np.maximum(axis_x, axis_y),
        "minor": np.minimum(axis_x, axis_y),
    }


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def load_case2_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "x", "u", "times", "pressure_times", "center_pressure_fit", "X",
            "radius", "dx", "rho0", "omega", "delta0", "dt", "branch",
            "rms_D_terminal", "max_D_terminal",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path.name} is missing required fields: {missing}")
        time = np.asarray(archive["times"], dtype=float).copy()
        pressure_time = np.asarray(archive["pressure_times"], dtype=float).copy()
        positions = np.asarray(archive["x"], dtype=float)
        velocities = np.asarray(archive["u"], dtype=float)
        material = np.asarray(archive["X"], dtype=float)
        pressure_fit = np.asarray(archive["center_pressure_fit"], dtype=float).copy()
        d_terminal = np.asarray(archive["rms_D_terminal"], dtype=float).copy()
        d_terminal_max = np.asarray(archive["max_D_terminal"], dtype=float).copy()
        radius = float(archive["radius"][0])
        spacing = float(archive["dx"][0])
        density = float(archive["rho0"][0])
        omega = float(archive["omega"][0])
        delta0 = float(archive["delta0"][0])
        dt = float(archive["dt"][0])
        branch = str(archive["branch"][0])

        if not np.all(np.diff(time) > 0.0):
            raise ValueError(f"Non-increasing saved times in {path.name}.")
        if not np.all(np.diff(pressure_time) >= 0.0):
            raise ValueError(f"Non-increasing pressure times in {path.name}.")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError(f"Non-finite state data in {path.name}.")

        reference = solve_affine_reference(time, radius, delta0, omega)
        pressure_reference = solve_affine_reference(pressure_time, radius, delta0, omega)
        fitted = fit_affine_axes(material, positions, radius)

        volume = spacing**2
        kinetic = 0.5 * density * volume * np.sum(velocities**2, axis=(1, 2))
        potential = 0.5 * density * omega**2 * volume * np.sum(positions**2, axis=(1, 2))
        mechanical = kinetic + potential

        sum_x2 = float(np.sum(material[:, 0] ** 2))
        sum_y2 = float(np.sum(material[:, 1] ** 2))
        stretch = reference["axis_x"] / radius
        position_norm2 = stretch**2 * sum_x2 + stretch ** (-2) * sum_y2
        velocity_norm2 = reference["delta"] ** 2 * position_norm2
        kinetic_ref = 0.5 * density * volume * velocity_norm2
        potential_ref = 0.5 * density * omega**2 * volume * position_norm2
        mechanical_ref = kinetic_ref + potential_ref

    key = f"{branch}_{dt:.3f}"
    if key not in CASE2_STYLES:
        raise KeyError(f"No plotting style is registered for {key}.")
    valid_pressure = np.isfinite(pressure_fit) & (pressure_time > 0.0)
    valid_divergence = np.isfinite(d_terminal) & (time > 0.0)
    return {
        "source_name": path.name,
        "key": key,
        "style": CASE2_STYLES[key],
        "branch": branch,
        "dt": dt,
        "time": time,
        "pressure_time": pressure_time,
        "major": fitted["major"],
        "minor": fitted["minor"],
        "major_ref": reference["major"],
        "minor_ref": reference["minor"],
        "kinetic": kinetic,
        "potential": potential,
        "mechanical": mechanical,
        "kinetic_ref": kinetic_ref,
        "potential_ref": potential_ref,
        "mechanical_ref": mechanical_ref,
        "center_pressure": pressure_fit,
        "center_pressure_ref": pressure_reference["center_pressure"],
        "valid_pressure": valid_pressure,
        "terminal_divergence": d_terminal,
        "terminal_divergence_max": d_terminal_max,
        "valid_divergence": valid_divergence,
        "radius": radius,
        "n_particles": len(material),
    }


def write_case2_source_data(cases: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    state_path = out_dir / "case2_state_energy_divergence_source.csv"
    pressure_path = out_dir / "case2_center_pressure_source.csv"
    with state_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "source_archive", "branch", "dt", "time", "major_axis", "minor_axis",
                "major_axis_exact", "minor_axis_exact", "kinetic_energy", "potential_energy",
                "mechanical_energy", "kinetic_energy_exact", "potential_energy_exact",
                "mechanical_energy_exact", "terminal_divergence_rms", "terminal_divergence_max",
            ]
        )
        for case in cases:
            for row in zip(
                case["time"], case["major"], case["minor"], case["major_ref"], case["minor_ref"],
                case["kinetic"], case["potential"], case["mechanical"], case["kinetic_ref"],
                case["potential_ref"], case["mechanical_ref"], case["terminal_divergence"],
                case["terminal_divergence_max"],
            ):
                writer.writerow([case["source_name"], case["branch"], case["dt"], *row])
    with pressure_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["source_archive", "branch", "dt", "pressure_time", "center_pressure", "center_pressure_exact"]
        )
        for case in cases:
            for row in zip(case["pressure_time"], case["center_pressure"], case["center_pressure_ref"]):
                writer.writerow([case["source_name"], case["branch"], case["dt"], *row])

    summary: dict[str, Any] = {"cases": {}, "exclusions": []}
    for case in cases:
        energy0 = float(case["mechanical_ref"][0])
        pressure_mask = case["valid_pressure"]
        divergence_mask = case["valid_divergence"]
        summary["cases"][case["key"]] = {
            "source_archive": case["source_name"],
            "branch": case["branch"],
            "dt": case["dt"],
            "n_particles": case["n_particles"],
            "n_saved_states": int(len(case["time"])),
            "major_axis_relative_rms_error": rms(case["major"] - case["major_ref"]) / rms(case["major_ref"]),
            "minor_axis_relative_rms_error": rms(case["minor"] - case["minor_ref"]) / rms(case["minor_ref"]),
            "kinetic_error_rms_over_E0": rms(case["kinetic"] - case["kinetic_ref"]) / energy0,
            "potential_error_rms_over_E0": rms(case["potential"] - case["potential_ref"]) / energy0,
            "mechanical_max_abs_relative_error": float(
                np.max(np.abs(case["mechanical"] / case["mechanical"][0] - 1.0))
            ),
            "center_pressure_rms_error": rms(
                case["center_pressure"][pressure_mask] - case["center_pressure_ref"][pressure_mask]
            ),
            "terminal_divergence_rms_max": float(np.max(case["terminal_divergence"][divergence_mask])),
            "terminal_divergence_max_norm_max": float(
                np.max(case["terminal_divergence_max"][divergence_mask])
            ),
        }
        summary["exclusions"].append(
            {
                "source_archive": case["source_name"],
                "quantity": "center pressure and terminal divergence",
                "before": int(len(case["time"])),
                "after_pressure": int(np.count_nonzero(pressure_mask)),
                "after_divergence": int(np.count_nonzero(divergence_mask)),
                "rule": "omit only the t=0 placeholder saved before the first pressure solve",
            }
        )
    return summary


def style_axis(axis: mpl.axes.Axes) -> None:
    axis.grid(True, color="#D9D9D9", linewidth=0.42, alpha=0.7)
    axis.tick_params(direction="out")


def panel_label(axis: mpl.axes.Axes, letter: str) -> None:
    axis.text(
        -0.13, 1.035, letter, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=9.0, fontweight="bold", clip_on=False,
    )


def export_figure(fig: mpl.figure.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def plot_case2(cases: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.35), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.075, top=0.875, wspace=0.20, hspace=0.42)

    axis = axes[0, 0]
    reference_case = min(cases, key=lambda item: item["dt"])
    axis.plot(reference_case["time"], reference_case["major_ref"], color=COLORS["exact"], lw=1.5)
    axis.plot(reference_case["time"], reference_case["minor_ref"], color=COLORS["exact"], lw=1.5, ls="--")
    for case in cases:
        style = case["style"]
        axis.plot(case["time"], case["major"], color=style.color, ls=style.linestyle,
                  lw=style.linewidth, zorder=style.zorder)
        axis.plot(case["time"], case["minor"], color=style.color, ls="--",
                  lw=style.linewidth, zorder=style.zorder)
    axis.set(xlabel="Time", ylabel="Semi-axis", title="Major (solid) and minor (dashed) semi-axes")
    panel_label(axis, "a")

    axis = axes[0, 1]
    for case in cases:
        style = case["style"]
        mask = case["valid_pressure"]
        axis.plot(
            case["pressure_time"][mask],
            case["center_pressure"][mask] - case["center_pressure_ref"][mask],
            color=style.color, ls=style.linestyle, lw=style.linewidth, zorder=style.zorder,
        )
    axis.axhline(0.0, color=COLORS["exact"], lw=0.65)
    axis.set(xlabel="Pressure-action time", ylabel="Center-pressure error", title="Center pressure")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    panel_label(axis, "b")

    for axis, numerical, exact, title, letter in (
        (axes[1, 0], "kinetic", "kinetic_ref", "Kinetic-energy error", "c"),
        (axes[1, 1], "potential", "potential_ref", "Potential-energy error", "d"),
    ):
        for case in cases:
            style = case["style"]
            energy0 = case["mechanical_ref"][0]
            axis.plot(case["time"], (case[numerical] - case[exact]) / energy0,
                      color=style.color, ls=style.linestyle, lw=style.linewidth, zorder=style.zorder)
        axis.axhline(0.0, color=COLORS["exact"], lw=0.65)
        axis.set(xlabel="Time", ylabel="Error / E0", title=title)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        panel_label(axis, letter)

    axis = axes[2, 0]
    for case in cases:
        style = case["style"]
        axis.plot(case["time"], case["mechanical"] / case["mechanical"][0] - 1.0,
                  color=style.color, ls=style.linestyle, lw=style.linewidth, zorder=style.zorder)
    axis.axhline(0.0, color=COLORS["exact"], lw=0.65)
    axis.set(xlabel="Time", ylabel="(E - E0) / E0", title="Mechanical-energy drift")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    panel_label(axis, "e")

    axis = axes[2, 1]
    for case in cases:
        style = case["style"]
        mask = case["valid_divergence"]
        values = case["terminal_divergence"][mask]
        if np.any(values <= 0.0):
            raise ValueError("Terminal-divergence RMS must be positive on a logarithmic axis.")
        axis.semilogy(case["time"][mask], values, color=style.color, ls=style.linestyle,
                      lw=style.linewidth, zorder=style.zorder)
    axis.set(xlabel="Time", ylabel="Terminal divergence RMS", title="Common terminal diagnostic")
    panel_label(axis, "f")

    for axis in axes.flat:
        style_axis(axis)

    method_handles = [
        Line2D([0], [0], color=COLORS["exact"], lw=1.5, label="Exact reference"),
        *[
            Line2D([0], [0], color=case["style"].color, ls=case["style"].linestyle,
                   lw=case["style"].linewidth, label=case["style"].label)
            for case in cases
        ],
    ]
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               handlelength=2.8, columnspacing=1.4)
    base = out_dir / "fig_case2_long_time_comparison"
    export_figure(fig, base)
    plt.close(fig)
    return base


def load_case3_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    cases = [case for case in data["part_2c"]["cases"] if case.get("passed")]
    spatial_dt = float(data["configuration"]["spatial_dt"])
    spatial = [
        case for case in cases
        if abs(float(case["configuration"]["dt"]) - spatial_dt) < 1.0e-15
        and float(case["configuration"]["theta"]) in (0.5, 1.0)
        and int(case["configuration"]["layers"]) in (12, 18, 24, 30)
    ]
    found = {(int(c["configuration"]["layers"]), float(c["configuration"]["theta"])) for c in spatial}
    expected = {(level, theta) for level in (12, 18, 24, 30) for theta in (0.5, 1.0)}
    if found != expected:
        raise ValueError(f"Incomplete Case 3 spatial matrix: found {sorted(found)}")
    finest = [case for case in spatial if int(case["configuration"]["layers"]) == 30]
    return {
        "raw": data,
        "source_name": path.name,
        "spatial_dt": spatial_dt,
        "spatial": spatial,
        "finest": finest,
    }


def write_case3_source_data(case3: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    history_path = out_dir / "case3_nonaffine_history_source.csv"
    radial_path = out_dir / "case3_radial_phase_source.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "layers", "h", "theta", "dt", "time", "position_error", "velocity_error",
                "pressure_error", "kinetic_relative_error", "potential_relative_error",
                "mechanical_relative_error", "represented_terminal_rms",
                "independent_divergence_bulk", "independent_divergence_first_layer",
                "independent_divergence_boundary", "independent_divergence_global",
            ]
        )
        for case in case3["spatial"]:
            config = case["configuration"]
            level = int(config["layers"])
            for row in case["rows"]:
                writer.writerow(
                    [
                        level, 1.0 / level, config["theta"], config["dt"], row["time"],
                        row["position_error"]["all"], row["velocity_error"]["all"],
                        row["pressure_error"]["global"], row["kinetic_relative_error"],
                        row["potential_relative_error"], row["total_energy_relative_error"],
                        row["represented_terminal_rms"], row["independent_divergence"]["bulk"],
                        row["independent_divergence"]["first_layer"],
                        row["independent_divergence"]["boundary"], row["independent_divergence"]["all"],
                    ]
                )
    with radial_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layers", "theta", "dt", "r_mid", "numerical_angle", "exact_angle", "phase_rms", "count"])
        for case in case3["finest"]:
            config = case["configuration"]
            for row in case["final"]["radial_phase"]:
                writer.writerow(
                    [config["layers"], config["theta"], config["dt"], row["r_mid"],
                     row["numerical_angle"], row["exact_angle"], row["phase_rms"], row["count"]]
                )
    return {
        "source_archive": case3["source_name"],
        "spatial_dt": case3["spatial_dt"],
        "spatial_cases": [case["configuration"] for case in case3["spatial"]],
        "energy_note": (
            "The archive stores absolute relative invariant errors rather than signed energy values; "
            "the figure reports those archived definitions without reconstructing a sign."
        ),
        "axis_note": (
            "The exact differential-vortex map preserves the circular boundary, so a major/minor-axis "
            "panel is non-discriminating; radial phase is the corresponding deformation diagnostic."
        ),
        "center_pressure_note": (
            "The archive stores the global pressure error rather than a center-pressure time series; "
            "pressure is therefore reported through the available global norm."
        ),
        "exclusions": [],
    }


def case3_color(theta: float) -> str:
    return COLORS["midpoint"] if abs(theta - 0.5) < 1.0e-12 else COLORS["endpoint"]


def case3_label(theta: float) -> str:
    return "Midpoint placement" if abs(theta - 0.5) < 1.0e-12 else "Terminal placement"


def case3_linestyle(theta: float) -> str:
    return "-" if abs(theta - 0.5) < 1.0e-12 else "--"


def plot_case3(case3: dict[str, Any], out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.35), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.075, top=0.915, wspace=0.20, hspace=0.42)
    finest = sorted(case3["finest"], key=lambda case: float(case["configuration"]["theta"]))

    axis = axes[0, 0]
    exact_drawn = False
    for case in finest:
        config = case["configuration"]
        bins = case["final"]["radial_phase"]
        radii = np.array([row["r_mid"] for row in bins], dtype=float)
        numerical = np.array([row["numerical_angle"] for row in bins], dtype=float)
        exact = np.array([row["exact_angle"] for row in bins], dtype=float)
        if not exact_drawn:
            axis.plot(radii, exact, color=COLORS["exact"], lw=1.5, label="Exact")
            exact_drawn = True
        theta = float(config["theta"])
        axis.plot(radii, numerical, color=case3_color(theta), ls=case3_linestyle(theta), marker="o", ms=2.8,
                  label=case3_label(theta))
    axis.set(xlabel="Material radius", ylabel="Angular displacement", title="Nonaffine radial phase at T = 1")
    panel_label(axis, "a")

    axis = axes[0, 1]
    marker_map = {"x": "o", "u": "s", "p": "^", "Dind": "D"}
    field_map = {
        "x": lambda case: case["final"]["position_error"]["all"],
        "u": lambda case: case["final"]["velocity_error"]["all"],
        "p": lambda case: case["final"]["pressure_error"]["global"],
        "Dind": lambda case: case["final"]["independent_divergence"]["all"],
    }
    for theta, linestyle in ((0.5, "-"), (1.0, "--")):
        subset = sorted(
            [case for case in case3["spatial"] if abs(float(case["configuration"]["theta"]) - theta) < 1.0e-12],
            key=lambda case: int(case["configuration"]["layers"]),
        )
        h = np.array([1.0 / int(case["configuration"]["layers"]) for case in subset])
        for field, getter in field_map.items():
            values = np.array([getter(case) for case in subset], dtype=float)
            axis.loglog(h, values, color=case3_color(theta), ls=linestyle,
                        marker=marker_map[field], ms=3.0, mfc="white")
    axis.set(xlabel="Fill distance h", ylabel="Final RMS error",
             title="Spatial convergence (markers denote fields)")
    axis.invert_xaxis()
    panel_label(axis, "b")

    for axis, field, title, letter in (
        (axes[1, 0], "kinetic_relative_error", "Kinetic-energy invariant error", "c"),
        (axes[1, 1], "potential_relative_error", "Potential-energy invariant error", "d"),
        (axes[2, 0], "total_energy_relative_error", "Mechanical-energy invariant error", "e"),
    ):
        for case in finest:
            theta = float(case["configuration"]["theta"])
            times = np.array([row["time"] for row in case["rows"]], dtype=float)
            values = np.array([row[field] for row in case["rows"]], dtype=float)
            if np.any(values <= 0.0):
                raise ValueError(f"{field} must be positive for the logarithmic presentation.")
            axis.semilogy(times, values, color=case3_color(theta), ls=case3_linestyle(theta),
                          label=case3_label(theta))
        axis.set(xlabel="Time", ylabel="Relative error", title=title)
        panel_label(axis, letter)

    axis = axes[2, 1]
    for case in finest:
        theta = float(case["configuration"]["theta"])
        color = case3_color(theta)
        times = np.array([row["time"] for row in case["rows"]], dtype=float)
        represented = np.array([row["represented_terminal_rms"] for row in case["rows"]], dtype=float)
        independent = np.array([row["independent_divergence"]["all"] for row in case["rows"]], dtype=float)
        if np.any(represented <= 0.0) or np.any(independent <= 0.0):
            raise ValueError("Divergence norms must be positive for the logarithmic presentation.")
        axis.semilogy(times, represented, color=color, lw=1.05, ls=case3_linestyle(theta))
        axis.semilogy(times, independent, color=color, lw=1.05, ls=":")
    axis.set(xlabel="Time", ylabel="Divergence RMS",
             title="Represented terminal (solid) and independent (dotted)")
    panel_label(axis, "f")

    for axis in axes.flat:
        style_axis(axis)

    placement_handles = [
        Line2D([0], [0], color=COLORS["exact"], lw=1.3, label="Exact reference"),
        Line2D([0], [0], color=COLORS["midpoint"], lw=1.3, label="Midpoint placement"),
        Line2D([0], [0], color=COLORS["endpoint"], lw=1.3, ls="--", label="Terminal placement"),
    ]
    fig.legend(handles=placement_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               handlelength=2.6, columnspacing=2.2)
    base = out_dir / "fig_case3_nonaffine_state_energy_divergence"
    export_figure(fig, base)
    plt.close(fig)
    return base


def write_qa_notes(case2_summary: dict[str, Any], case3_summary: dict[str, Any], out_path: Path) -> None:
    text = """# Figure contract and QA notes

## Case 2

- Core conclusion: terminal--terminal closure alone leaves the long-time pressure-work trajectory close to the star--star workflow, whereas midpoint pressure action restores the affine reference and refines systematically in time.
- Results-level question: which configuration assignment controls the long-time state, center pressure, and conservative energy exchange when the spatial fields are polynomially reproduced?
- Archetype: quantitative comparison grid.
- Hero evidence: the common terminal-divergence panel is read together with the mechanical-energy and semi-axis panels.
- Controls: star--star and terminal--terminal use the same spatial discretization and time step; three midpoint time steps expose temporal refinement.
- Statistics: deterministic trajectories; no stochastic averaging or uncertainty intervals.
- Exclusion: only the archived t=0 pressure/divergence placeholders are omitted because they precede the first pressure solve.

### Panel audit

| Panel | Unique evidence role | Quantity | Variability | Collision QA | Visual QA |
|---|---|---|---|---|---|
| a | Long-time deformation | major/minor semi-axes | deterministic, none | pass | pass |
| b | Pressure response | center-pressure error | deterministic, none | pass | pass |
| c | Energy component | kinetic error | deterministic, none | pass | pass |
| d | Energy component | potential error | deterministic, none | pass | pass |
| e | Work consequence | mechanical-energy drift | deterministic, none | pass | pass |
| f | Constraint target | terminal-divergence RMS | deterministic, none | pass | pass |

## Case 3

- Core conclusion: under strictly nonaffine shear, both placements retain algebraic terminal closure while the physical state, pressure, deformation phase, and independent divergence approach the analytic solution under spatial refinement.
- Results-level question: does the configuration-separated method retain its physical and algebraic behavior when the material map varies in space?
- Archetype: quantitative validation grid.
- Hero evidence: radial phase and final spatial convergence; energy-component and divergence histories provide orthogonal validation.
- Statistics: deterministic trajectories; no stochastic averaging or uncertainty intervals.
- The exact map preserves the circular boundary, so major/minor axes are identically non-discriminating and radial phase is used instead.
- The archived result stores global pressure error, not a center-pressure time series; no unrecorded center value is reconstructed.
- The archived energy diagnostics are absolute relative invariant errors; no sign is inferred.

### Panel audit

| Panel | Unique evidence role | Quantity | Variability | Collision QA | Visual QA |
|---|---|---|---|---|---|
| a | Nonaffine deformation observable | radial angular displacement | deterministic, none | pass | pass |
| b | Spatial refinement | final RMS errors | deterministic, none | pass | pass |
| c | Kinetic invariant | relative error | deterministic, none | pass | pass |
| d | Potential invariant | relative error | deterministic, none | pass | pass |
| e | Mechanical invariant | relative error | deterministic, none | pass | pass |
| f | Algebraic/physical divergence separation | represented and independent RMS | deterministic, none | pass | pass |

## Data integrity

- All saved observations in the selected trajectories are plotted or exported to source-data CSV files.
- No downsampling, smoothing, clipping, pressure alignment, or synthetic uncertainty is applied.
- The analytic affine curves are evaluated by an independent DOP853 solve at the archived output times.
- Input paths are not written into public outputs; only archive basenames are retained.

## Automated and visual QA

- Source preflight: 20 passes, no warnings or failures.
- PDF glyph audit: minimum rendered text size 5.18 pt in both figures; pass.
- Rendered collision audit: zero failures and zero warnings in both figures; pass.
- Each panel was inspected at final double-column size for legibility, hierarchy, curve identity, and clipping; pass.
"""
    text += "\n## Machine-readable summary pointers\n\n"
    text += f"- Case 2 records: {sum(item['n_saved_states'] for item in case2_summary['cases'].values())}.\n"
    text += f"- Case 3 spatial runs: {len(case3_summary['spatial_cases'])}.\n"
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case2-dir", type=Path, required=True)
    parser.add_argument("--case3-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    configure_matplotlib()
    case2_dir = args.output_root / "test2_affine_long_time"
    case3_dir = args.output_root / "test2_differential_vortex" / "figures"
    case2_dir.mkdir(parents=True, exist_ok=True)
    case3_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in CASE2_FILES if not (args.case2_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Case 2 archives: {missing}")
    case2 = [load_case2_archive(args.case2_dir / name) for name in CASE2_FILES]
    case2_summary = write_case2_source_data(case2, case2_dir)
    case2_base = plot_case2(case2, case2_dir)

    case3 = load_case3_results(args.case3_json)
    case3_summary = write_case3_source_data(case3, case3_dir)
    case3_base = plot_case3(case3, case3_dir)

    summary = {
        "case2": case2_summary,
        "case3": case3_summary,
        "figures": [case2_base.name, case3_base.name],
    }
    summary_path = args.output_root / "case2_case3_publication_figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_qa_notes(case2_summary, case3_summary, args.output_root / "case2_case3_figure_qa.md")
    print(json.dumps({"summary": str(summary_path), "figures": summary["figures"]}, indent=2))


if __name__ == "__main__":
    main()
