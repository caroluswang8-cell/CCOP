"""Constant-field and frozen translation audit for the nonaffine CCOP map.

This script checks a necessary physical-consistency property that is not
implied by recursive/direct operator agreement.  It uses the exact prescribed
differential-vortex map at chi=1 and the production midpoint action path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from test2_differential_vortex import (
    build_discretization,
    exact_map,
    production_path_position,
    relative_geometry,
    transport_action_sparse,
    transport_divergence_sparse,
)


LAYERS = (12, 18, 24, 30)
CHI = 1.0
THETA = 0.5
OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "test2_differential_vortex"
    / "constant_translation_audit.json"
)


def weighted_scalar_rms(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weight * value**2) / np.sum(weight)))


def weighted_vector_rms(value: np.ndarray, weight: np.ndarray) -> float:
    return float(
        np.sqrt(np.sum(weight * np.sum(value**2, axis=1)) / np.sum(weight))
    )


def fitted_rate(values: list[float]) -> float:
    h = 1.0 / np.asarray(LAYERS, dtype=np.float64)
    return float(np.polyfit(np.log(h), np.log(values), 1)[0])


def audit_level(layers: int) -> dict:
    disc = build_discretization(layers)
    particle_count = len(disc.points)
    identity = np.broadcast_to(np.eye(2), (particle_count, 2, 2))
    terminal = relative_geometry(
        disc.derivative,
        exact_map(disc.points, CHI) - disc.points,
        identity,
    )
    action_geometry = relative_geometry(
        disc.derivative,
        production_path_position(disc.points, CHI, THETA) - disc.points,
        identity,
    )
    divergence = transport_divergence_sparse(
        disc.D_q0,
        terminal["inverse"],
        terminal["J"],
        disc.interior,
    )
    action = transport_action_sparse(
        disc.G0,
        action_geometry["inverse"],
        action_geometry["J"],
    )
    response = (divergence @ action).tocsc()

    records = []
    for component in range(2):
        constant = np.zeros((particle_count, 2), dtype=np.float64)
        constant[:, component] = 1.0
        divergence_value = np.asarray(divergence @ constant.ravel())
        pressure = spla.spsolve(response, divergence_value)
        correction = np.asarray(action @ pressure).reshape(-1, 2)
        relative_residual = float(
            np.linalg.norm(response @ pressure - divergence_value)
            / (np.linalg.norm(divergence_value) + 1.0e-30)
        )
        records.append(
            {
                "component": component,
                "constant_divergence_rms": weighted_scalar_rms(
                    divergence_value, disc.volume[disc.interior]
                ),
                "constant_divergence_max": float(
                    np.max(np.abs(divergence_value))
                ),
                "frozen_translation_projection_rms": weighted_vector_rms(
                    correction, disc.volume
                ),
                "linear_relative_residual": relative_residual,
            }
        )

    return {
        "layers": layers,
        "particles": particle_count,
        "pressure_dofs": len(disc.interior),
        "minimum_terminal_J": float(np.min(terminal["J"])),
        "minimum_action_J": float(np.min(action_geometry["J"])),
        "components": records,
        "maximum_constant_divergence_rms": max(
            item["constant_divergence_rms"] for item in records
        ),
        "maximum_translation_projection_rms": max(
            item["frozen_translation_projection_rms"] for item in records
        ),
    }


def main() -> None:
    rows = [audit_level(layers) for layers in LAYERS]
    result = {
        "configuration": {
            "layers": LAYERS,
            "chi": CHI,
            "theta": THETA,
            "map": "exact prescribed differential vortex",
            "solve": "square pressure-test response with sparse LU",
            "regularization": False,
            "extra_projection": False,
        },
        "rows": rows,
        "fitted_rates": {
            "constant_divergence_rms": fitted_rate(
                [row["maximum_constant_divergence_rms"] for row in rows]
            ),
            "frozen_translation_projection_rms": fitted_rate(
                [row["maximum_translation_projection_rms"] for row in rows]
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
