"""Small time-integration utilities shared by the manuscript drivers."""

from __future__ import annotations

import numpy as np


def implicit_midpoint_force_predictor(
    points: np.ndarray,
    velocity: np.ndarray,
    force_matrix: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pressure-free midpoint predictor and position increment.

    The conservative force is ``f(x) = -force_matrix @ x``.  The formula is
    algebraically identical to the predictor used in the validated source
    driver; it is isolated here so Test 2 does not import unrelated legacy
    experiment modules.
    """

    matrix = np.eye(2, dtype=np.float64) + 0.25 * dt * dt * force_matrix
    rhs = velocity - 0.5 * dt * (points @ force_matrix.T)
    midpoint_velocity = np.linalg.solve(matrix, rhs.T).T
    velocity_star = 2.0 * midpoint_velocity - velocity
    increment_star = 0.5 * dt * (velocity + velocity_star)
    return velocity_star, increment_star

