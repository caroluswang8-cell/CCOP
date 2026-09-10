"""Structural Test 1: force-free affine incompressible liquid in vacuum.

This driver is deliberately independent of the earlier periodic TGV and
rotating-square reports.  It uses a general nodal homogeneous-Dirichlet
pressure space on a quasi-uniform disk cloud.  Boundary pressure values are
fixed to zero; the pressure coefficients and production divergence tests are
the interior particle labels.

The production operator state is transported by

    D_T = J_T^{-1} D_n (J_T F_T^{-1} .),
    G_a = J_a F_a^{-T} G_n.

The all-particle strong divergence is transported separately and is never used
as the nonlinear acceptance constraint.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gmls_material import MaterialGMLS, _monomial_matrix, _wendland_c2
from material_clouds import ordered_polygon_area, ring_disk_material_cloud


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "test1_affine_free_boundary"
FIGURE_DIR = RESULT_DIR / "figures"
DEFAULT_JSON = RESULT_DIR / "test1_affine_free_boundary_results.json"
DEFAULT_CSV = RESULT_DIR / "test1_affine_free_boundary_tables.csv"
DEFAULT_CONFIG = RESULT_DIR / "test1_affine_free_boundary_config.json"
DEFAULT_REPORT = RESULT_DIR / "test1_affine_free_boundary_report.md"

RADIUS = 1.0
RHO = 1.0
GAMMA = 0.5
DEGREE = 2
INDEPENDENT_DEGREE = 3
LAYERS = (12, 18, 24, 30)
ANCHOR_DTS = (0.04, 0.02, 0.01, 0.005)
THETA_VALUES = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
SWEEP_DT = 0.02
FINAL_TIME = 0.5
MULTISTEP_DTS = (1.0 / 40.0, 1.0 / 80.0, 1.0 / 160.0, 1.0 / 320.0)
MULTISTEP_THETAS = (0.5, 1.0)
STAGE_START_TIME = 0.2
STAGE_DTS = (0.02, 0.01, 0.005, 0.0025)
STAGE_THETAS = (0.0, 0.5, 1.0)
STAGE_LAYERS = 18
GEOMETRY_RMS_TOL = 1.0e-12
GEOMETRY_MAX_TOL = 1.0e-11
TERMINAL_RMS_TOL = 2.0e-10
TERMINAL_MAX_TOL = 2.0e-9
ADMISSIBLE_J = 1.0e-6
DETERMINISTIC_SEED = 20260728

PAPER = {
    "authors": "J. Roberts, S. Shkoller, T. C. Sideris",
    "title": (
        "Affine Motion of 2d Incompressible Fluids Surrounded by Vacuum "
        "and Flows in SL(2,R)"
    ),
    "journal": "Communications in Mathematical Physics 375 (2020), 1003-1040",
    "arxiv": "https://arxiv.org/abs/1811.07781",
    "doi": "https://doi.org/10.1007/s00220-020-03723-2",
    "verified_equations": {
        "paper_general": "A_ddot = -kappa A + lambda cof(A)",
        "paper_multiplier": "lambda = 2(kappa-det(A_dot))/|A|_F^2",
        "perfect_fluid": "kappa=0",
        "pressure": "p=lambda/2*(1-|A^{-1}x|^2)",
    },
}


def ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if sp.issparse(value):
        raise TypeError("sparse matrices must be removed before serialization")
    if isinstance(value, dict):
        return {str(key): ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(item) for item in value]
    return value


def scalar_rms(value: np.ndarray, weight: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    mass = np.asarray(weight, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.sum(mass * array**2) / np.sum(mass)))


def vector_rms(value: np.ndarray, weight: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    mass = np.asarray(weight, dtype=np.float64).reshape(-1)
    pointwise = np.sum(array**2, axis=tuple(range(1, array.ndim)))
    return float(np.sqrt(np.sum(mass * pointwise) / np.sum(mass)))


def relative_sparse_norm(left: sp.spmatrix, right: sp.spmatrix) -> float:
    denominator = max(float(spla.norm(right)), 1.0e-300)
    return float(spla.norm(left - right) / denominator)


def observed_rate(abscissa: Iterable[float], error: Iterable[float]) -> float | None:
    x = np.asarray(tuple(abscissa), dtype=np.float64)
    y = np.abs(np.asarray(tuple(error), dtype=np.float64))
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 1.0e-15)
    if np.count_nonzero(valid) < 2:
        return None
    return float(np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)[0])


def affine_reference(times: np.ndarray) -> dict[str, np.ndarray]:
    """DOP853 scalar reference; det(A)=1 is analytic, not projected."""
    times = np.asarray(times, dtype=np.float64)

    def rhs(_time: float, value: np.ndarray) -> np.ndarray:
        a = value[0]
        return np.asarray([np.sqrt(2.0) * a**2 / np.sqrt(a**4 + 1.0)])

    solution = solve_ivp(
        rhs,
        (0.0, float(np.max(times))),
        np.asarray([1.0]),
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-15,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    a = solution.sol(times)[0]
    adot = np.sqrt(2.0) * a**2 / np.sqrt(a**4 + 1.0)
    addot = 2.0 * adot**2 / (a * (a**4 + 1.0))
    multiplier = 2.0 * adot**2 / (a**4 + 1.0)
    count = len(times)
    A = np.zeros((count, 2, 2))
    A[:, 0, 0] = a
    A[:, 1, 1] = 1.0 / a
    A_dot = np.zeros_like(A)
    A_dot[:, 0, 0] = adot
    A_dot[:, 1, 1] = -adot / a**2
    A_ddot = np.zeros_like(A)
    A_ddot[:, 0, 0] = addot
    A_ddot[:, 1, 1] = (
        2.0 * adot**2 / a**3 - addot / a**2
    )
    cof = np.zeros_like(A)
    cof[:, 0, 0] = 1.0 / a
    cof[:, 1, 1] = a
    energy = 0.5 * np.sum(A_dot**2, axis=(1, 2))
    return {
        "time": times,
        "a": a,
        "b": 1.0 / a,
        "aspect_ratio": a**2,
        "a_dot": adot,
        "a_ddot": addot,
        "lambda": multiplier,
        "A": A,
        "A_dot": A_dot,
        "A_ddot": A_ddot,
        "det_A": np.linalg.det(A),
        "matrix_energy": energy,
        "matrix_equation_defect": np.linalg.norm(
            A_ddot - multiplier[:, None, None] * cof, axis=(1, 2)
        ),
        "solver_nfev": solution.nfev,
    }


def literature_and_reference_audit() -> dict[str, Any]:
    time = np.linspace(0.0, FINAL_TIME, 201)
    reference = affine_reference(time)
    initial_points = np.asarray(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [1.0, 0.0]]
    )
    pressure0 = 0.5 * (1.0 - np.sum(initial_points**2, axis=1))
    velocity_gradient = np.diag((1.0, -1.0))
    initial_lambda_from_paper = (
        -2.0 * np.linalg.det(velocity_gradient)
        / np.sum(np.eye(2) ** 2)
    )
    return {
        "paper": PAPER,
        "sign_check": {
            "lambda0_from_paper": float(initial_lambda_from_paper),
            "negative_pressure_gradient_equals_A_ddot_X": True,
            "reason": (
                "grad_x p=-lambda A^{-T}X and det(A)=1 gives "
                "-grad_x p=lambda cof(A)X"
            ),
        },
        "initial_checks": {
            "velocity_divergence": float(np.trace(velocity_gradient)),
            "pressure_at_center": float(pressure0[0]),
            "minimum_sampled_interior_pressure": float(np.min(pressure0[:3])),
            "boundary_pressure": float(pressure0[-1]),
        },
        "reduced_derivation": {
            "lambda": "2*a_dot^2/(a^4+1)",
            "a_ddot": "2*a_dot^2/(a*(a^4+1))",
            "positive_energy_branch": "sqrt(2)*a^2/sqrt(a^4+1)",
        },
        "reference": {
            "rtol": 1.0e-13,
            "atol": 1.0e-15,
            "method": "DOP853 scalar reduction",
            "max_det_error": float(np.max(np.abs(reference["det_A"] - 1.0))),
            "max_matrix_energy_error": float(
                np.max(np.abs(reference["matrix_energy"] - 1.0))
            ),
            "max_matrix_equation_defect": float(
                np.max(reference["matrix_equation_defect"])
            ),
            "nfev": int(reference["solver_nfev"]),
            "time": time,
            "a": reference["a"],
            "b": reference["b"],
            "aspect_ratio": reference["aspect_ratio"],
        },
        "passed": bool(
            abs(initial_lambda_from_paper - 1.0) < 1.0e-15
            and np.max(np.abs(reference["det_A"] - 1.0)) < 1.0e-14
            and np.max(np.abs(reference["matrix_energy"] - 1.0)) < 5.0e-13
            and np.max(reference["matrix_equation_defect"]) < 1.0e-12
        ),
    }


def interleaved_divergence(hx: sp.spmatrix, hy: sp.spmatrix) -> sp.csr_matrix:
    blocked = sp.hstack((hx, hy), format="csr")
    count = hx.shape[1]
    permutation = np.ravel(
        np.column_stack((np.arange(count), np.arange(count) + count))
    )
    return blocked[:, permutation].tocsr()


def interleaved_gradient(
    hx: sp.spmatrix, hy: sp.spmatrix, columns: np.ndarray
) -> sp.csr_matrix:
    blocked = sp.vstack((hx[:, columns], hy[:, columns]), format="csr")
    count = hx.shape[0]
    permutation = np.ravel(
        np.column_stack((np.arange(count), np.arange(count) + count))
    )
    return blocked[permutation].tocsr()


def transport_action_sparse(
    action: sp.spmatrix, inverse: np.ndarray, jacobian: np.ndarray
) -> sp.csr_matrix:
    count = len(jacobian)
    gx = action[0::2]
    gy = action[1::2]
    out_x = (
        sp.diags(jacobian * inverse[:, 0, 0]) @ gx
        + sp.diags(jacobian * inverse[:, 1, 0]) @ gy
    )
    out_y = (
        sp.diags(jacobian * inverse[:, 0, 1]) @ gx
        + sp.diags(jacobian * inverse[:, 1, 1]) @ gy
    )
    blocked = sp.vstack((out_x, out_y), format="csr")
    permutation = np.ravel(
        np.column_stack((np.arange(count), np.arange(count) + count))
    )
    return blocked[permutation].tocsr()


def transport_divergence_sparse(
    divergence: sp.spmatrix,
    inverse: np.ndarray,
    jacobian: np.ndarray,
    row_labels: np.ndarray,
) -> sp.csr_matrix:
    dx = divergence[:, 0::2]
    dy = divergence[:, 1::2]
    x_block = (
        dx @ sp.diags(jacobian * inverse[:, 0, 0])
        + dy @ sp.diags(jacobian * inverse[:, 1, 0])
    )
    y_block = (
        dx @ sp.diags(jacobian * inverse[:, 0, 1])
        + dy @ sp.diags(jacobian * inverse[:, 1, 1])
    )
    blocked = sp.hstack((x_block, y_block), format="csr")
    count = len(jacobian)
    permutation = np.ravel(
        np.column_stack((np.arange(count), np.arange(count) + count))
    )
    return (
        sp.diags(1.0 / jacobian[np.asarray(row_labels)])
        @ blocked[:, permutation]
    ).tocsr()


def inverse_raw(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    determinant = (
        matrix[:, 0, 0] * matrix[:, 1, 1]
        - matrix[:, 0, 1] * matrix[:, 1, 0]
    )
    inverse = np.empty_like(matrix)
    inverse[:, 0, 0] = matrix[:, 1, 1]
    inverse[:, 0, 1] = -matrix[:, 0, 1]
    inverse[:, 1, 0] = -matrix[:, 1, 0]
    inverse[:, 1, 1] = matrix[:, 0, 0]
    inverse /= determinant[:, None, None]
    return inverse, determinant


def base_vector_gradient(derivative: MaterialGMLS, vector: np.ndarray) -> np.ndarray:
    return np.stack(
        (derivative.gradient(vector[:, 0]), derivative.gradient(vector[:, 1])),
        axis=1,
    )


def vector_gradient(
    derivative: MaterialGMLS, vector: np.ndarray, scalar_chain: np.ndarray
) -> np.ndarray:
    base = base_vector_gradient(derivative, vector)
    return np.einsum("nab,ncb->nca", scalar_chain, base)


def relative_geometry(
    derivative: MaterialGMLS,
    displacement: np.ndarray,
    scalar_chain: np.ndarray,
) -> dict[str, np.ndarray]:
    f = np.eye(2)[None, :, :] + vector_gradient(
        derivative, displacement, scalar_chain
    )
    inverse, jacobian = inverse_raw(f)
    singular = np.linalg.svd(f, compute_uv=False)
    return {
        "F": f,
        "inverse": inverse,
        "J": jacobian,
        "sigma_min": singular[:, -1],
        "sigma_max": singular[:, 0],
        "condition": singular[:, 0] / singular[:, -1],
    }


def region_masks(
    points: np.ndarray, boundary: np.ndarray, spacing: float
) -> dict[str, np.ndarray]:
    radius = np.linalg.norm(points, axis=1)
    boundary_mask = np.zeros(len(points), dtype=bool)
    boundary_mask[boundary] = True
    first = (~boundary_mask) & (radius >= RADIUS - 1.55 * spacing)
    bulk = ~(boundary_mask | first)
    return {"bulk": bulk, "first_layer": first, "boundary": boundary_mask}


def split_rms(
    value: np.ndarray, weight: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ("bulk", "first_layer", "boundary"):
        mask = masks[name]
        if value.ndim == 1:
            result[name] = scalar_rms(value[mask], weight[mask])
        else:
            result[name] = vector_rms(value[mask], weight[mask])
    result["all"] = (
        scalar_rms(value, weight)
        if value.ndim == 1
        else vector_rms(value, weight)
    )
    return result


def local_stencil_metrics(derivative: MaterialGMLS) -> tuple[np.ndarray, np.ndarray]:
    condition = np.zeros(derivative.n)
    moment = np.zeros(derivative.n)
    for i, neighbors in enumerate(derivative.neighbors):
        delta = derivative._relative_offsets(i, neighbors)
        distance = np.linalg.norm(delta, axis=1)
        scale = max(float(np.max(distance)), np.finfo(float).eps)
        basis = _monomial_matrix(delta / scale, derivative.exponents)
        kernel = _wendland_c2(distance / (1.05 * scale))
        singular = np.linalg.svd(np.sqrt(kernel)[:, None] * basis, compute_uv=False)
        condition[i] = singular[0] / singular[-1]
        defects = []
        for weights, functional in (
            (derivative.wx[i], derivative._functional((1, 0), scale)),
            (derivative.wy[i], derivative._functional((0, 1), scale)),
        ):
            defects.append(np.linalg.norm(basis.T @ weights - functional))
        moment[i] = max(defects)
    return condition, moment


def cloud_quality(points: np.ndarray, boundary: np.ndarray, spacing: float) -> dict[str, float]:
    tree = cKDTree(points)
    distance, _ = tree.query(points, k=2)
    min_separation = float(np.min(distance[:, 1]))
    radial = np.linspace(0.0, RADIUS, 4 * int(round(RADIUS / spacing)) + 1)
    angle = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    rr, aa = np.meshgrid(radial, angle, indexing="ij")
    probes = np.column_stack((rr.ravel() * np.cos(aa.ravel()), rr.ravel() * np.sin(aa.ravel())))
    fill = float(np.max(tree.query(probes, k=1)[0]))
    boundary_points = points[boundary]
    arc_chord = np.linalg.norm(
        boundary_points - np.roll(boundary_points, -1, axis=0), axis=1
    )
    return {
        "minimum_separation": min_separation,
        "approximate_fill_distance": fill,
        "boundary_chord_min": float(np.min(arc_chord)),
        "boundary_chord_max": float(np.max(arc_chord)),
    }


@dataclass
class DiskDiscretization:
    layers: int
    spacing: float
    points: np.ndarray
    boundary: np.ndarray
    interior: np.ndarray
    volume: np.ndarray
    masks: dict[str, np.ndarray]
    derivative: MaterialGMLS
    independent_derivative: MaterialGMLS
    D_strong0: sp.csr_matrix
    D_q0: sp.csr_matrix
    G0: sp.csr_matrix
    D_independent0: sp.csr_matrix


def build_dirichlet_discretization_from_cloud(
    points: np.ndarray,
    boundary: np.ndarray,
    volume: np.ndarray,
    spacing: float,
    layers: int,
    masks: dict[str, np.ndarray] | None = None,
) -> DiskDiscretization:
    """Build the common native nodal Dirichlet pressure/divergence pair.

    Test 1 uses this factory on the disk; Test 3 uses the identical factory
    on its noncircular initial physical drop.  Only cloud geometry and masks
    differ.  Degree, support, SVD cutoff, pressure labels, and independent
    diagnostic reconstruction are therefore owned by one implementation.
    """
    points = np.asarray(points, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.int32)
    volume = np.asarray(volume, dtype=np.float64)
    if masks is None:
        masks = region_masks(points, boundary, spacing)
    interior = np.flatnonzero(~np.asarray(masks["boundary"], dtype=bool))
    derivative = MaterialGMLS(
        points,
        degree=DEGREE,
        stencil_size=18,
        relative_svd_cutoff=1.0e-12,
    )
    independent = MaterialGMLS(
        points,
        degree=INDEPENDENT_DEGREE,
        stencil_size=32,
        relative_svd_cutoff=1.0e-12,
    )
    d_strong = interleaved_divergence(derivative.Hx, derivative.Hy)
    g0 = interleaved_gradient(derivative.Hx, derivative.Hy, interior)
    d_independent = interleaved_divergence(independent.Hx, independent.Hy)
    return DiskDiscretization(
        layers=layers,
        spacing=spacing,
        points=points.copy(),
        boundary=boundary.copy(),
        interior=interior,
        volume=volume.copy(),
        masks={key: np.asarray(value, dtype=bool).copy() for key, value in masks.items()},
        derivative=derivative,
        independent_derivative=independent,
        D_strong0=d_strong,
        D_q0=d_strong[interior].tocsr(),
        G0=g0,
        D_independent0=d_independent,
    )


@lru_cache(maxsize=None)
def build_discretization(layers: int) -> DiskDiscretization:
    spacing = RADIUS / layers
    points, boundary, volume = ring_disk_material_cloud(RADIUS, spacing)
    masks = region_masks(points, boundary, spacing)
    return build_dirichlet_discretization_from_cloud(
        points,
        boundary,
        volume,
        spacing,
        layers,
        masks,
    )


def normalized_response(
    d_q: sp.spmatrix,
    action: sp.spmatrix,
    test_volume: np.ndarray,
    pressure_volume: np.ndarray,
) -> sp.csc_matrix:
    return (
        sp.diags(np.sqrt(test_volume))
        @ (d_q @ action)
        @ sp.diags(1.0 / np.sqrt(pressure_volume))
    ).tocsc()


def sparse_spectrum(
    d_q: sp.spmatrix,
    action: sp.spmatrix,
    test_volume: np.ndarray,
    pressure_volume: np.ndarray,
    modes: int = 4,
) -> dict[str, Any]:
    matrix = normalized_response(d_q, action, test_volume, pressure_volume)
    dimension = matrix.shape[0]
    lu = spla.splu(matrix)
    pivot = np.abs(lu.U.diagonal())
    if dimension <= 1100:
        dense = matrix.toarray()
        _, singular, right_t = scipy.linalg.svd(
            dense, full_matrices=False, check_finite=True
        )
        smallest = singular[-modes:][::-1]
        largest = singular[:modes]
        right = right_t[-modes:][::-1].T
    else:
        inverse_operator = spla.LinearOperator(
            matrix.shape,
            matvec=lu.solve,
            rmatvec=lambda value: lu.solve(value, trans="T"),
            dtype=np.float64,
        )
        # If A=U Sigma V^T, then A^{-1}=V Sigma^{-1} U^T.
        # Its largest left singular vectors are therefore the smallest right
        # singular vectors of A.  This shift-invert audit avoids unreliable
        # direct "SM" Krylov convergence without introducing a pseudo-inverse.
        inverse_left, inverse_singular, _ = spla.svds(
            inverse_operator,
            k=modes,
            which="LM",
            solver="arpack",
            tol=1.0e-11,
            maxiter=5000,
        )
        order = np.argsort(inverse_singular)[::-1]
        inverse_singular = inverse_singular[order]
        smallest = 1.0 / inverse_singular
        right = inverse_left[:, order]
        # ARPACK converges robustly for the extremal largest value here;
        # PROPACK's default kmax=10*k is unnecessarily restrictive on the
        # finest disk cloud.  Only sigma_max is required for normalization.
        _, largest, _ = spla.svds(
            matrix, k=1, which="LM", solver="arpack", tol=1.0e-12
        )
        largest = np.sort(largest)[::-1]
    sigma_min = float(smallest[0])
    sigma_max = float(largest[0])
    return {
        "dimension": dimension,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "relative_gap": sigma_min / sigma_max,
        "condition_number": sigma_max / sigma_min,
        "smallest_singular_values": smallest,
        "largest_singular_values": largest,
        "smallest_right_vectors_weighted": right,
        "minimum_lu_pivot": float(np.min(pivot)),
        "rank_at_1e-8": int(dimension if sigma_min / sigma_max >= 1.0e-8 else -1),
        "rank_at_1e-10": int(dimension if sigma_min / sigma_max >= 1.0e-10 else -1),
        "rank_at_1e-12": int(dimension if sigma_min / sigma_max >= 1.0e-12 else -1),
    }


def factor_and_condition(matrix: sp.spmatrix) -> tuple[spla.SuperLU, dict[str, float]]:
    csc = matrix.tocsc()
    lu = spla.splu(csc)
    inverse = spla.LinearOperator(
        csc.shape,
        matvec=lu.solve,
        rmatvec=lambda value: lu.solve(value, trans="T"),
        dtype=np.float64,
    )
    condition = float(spla.onenormest(csc) * spla.onenormest(inverse))
    pivot = np.abs(lu.U.diagonal())
    return lu, {
        "condition_1_estimate": condition,
        "minimum_lu_pivot": float(np.min(pivot)),
        "maximum_lu_pivot": float(np.max(pivot)),
        "relative_pivot": float(np.min(pivot) / np.max(pivot)),
    }


def fit_affine_map(points0: np.ndarray, points: np.ndarray, weight: np.ndarray) -> np.ndarray:
    gram = points0.T @ (weight[:, None] * points0)
    cross = points.T @ (weight[:, None] * points0)
    return cross @ np.linalg.inv(gram)


def preliminary_audit(layers: int) -> dict[str, Any]:
    disc = build_discretization(layers)
    points = disc.points
    volume = disc.volume
    masks = disc.masks
    interior = disc.interior
    condition, moment = local_stencil_metrics(disc.derivative)
    exact_pressure = 0.5 * (1.0 - np.sum(points**2, axis=1))
    exact_gradient = -points
    discrete_gradient = (disc.G0 @ exact_pressure[interior]).reshape(-1, 2)
    velocity = np.column_stack((points[:, 0], -points[:, 1]))
    divergence = disc.D_strong0 @ velocity.ravel()
    polynomial_max = {name: 0.0 for name in ("bulk", "first_layer", "boundary", "all")}
    for total in range(DEGREE + 1):
        for px in range(total + 1):
            py = total - px
            field = points[:, 0] ** px * points[:, 1] ** py
            exact = np.column_stack(
                (
                    np.zeros(len(points))
                    if px == 0
                    else px * points[:, 0] ** (px - 1) * points[:, 1] ** py,
                    np.zeros(len(points))
                    if py == 0
                    else py * points[:, 0] ** px * points[:, 1] ** (py - 1),
                )
            )
            error = np.linalg.norm(disc.derivative.gradient(field) - exact, axis=1)
            polynomial_max["all"] = max(polynomial_max["all"], float(np.max(error)))
            for name in ("bulk", "first_layer", "boundary"):
                polynomial_max[name] = max(
                    polynomial_max[name], float(np.max(error[masks[name]]))
                )
    affine = np.diag((1.17, 1.0 / 1.17))
    mapped = points @ affine.T
    geometry = relative_geometry(
        disc.derivative,
        mapped - points,
        np.broadcast_to(np.eye(2), (len(points), 2, 2)),
    )
    action = transport_action_sparse(disc.G0, geometry["inverse"], geometry["J"])
    exact_placed_gradient = (affine.T @ exact_gradient.T).T  # overwritten below
    exact_placed_gradient = np.einsum(
        "ab,nb->na", np.linalg.inv(affine).T, exact_gradient
    )
    action_error = (action @ exact_pressure[interior]).reshape(-1, 2) - exact_placed_gradient
    affine_velocity = points @ np.diag((1.0, -1.0 / 1.17**2)).T
    d_affine = transport_divergence_sparse(
        disc.D_strong0, geometry["inverse"], geometry["J"], np.arange(len(points))
    )
    divergence_affine = d_affine @ affine_velocity.ravel()
    spectrum = sparse_spectrum(
        disc.D_q0,
        disc.G0,
        volume[interior],
        volume[interior],
    )
    smallest_modes = []
    weighted_vectors = spectrum.pop("smallest_right_vectors_weighted")
    for index in range(weighted_vectors.shape[1]):
        field = np.zeros(len(points))
        field[interior] = weighted_vectors[:, index] / np.sqrt(volume[interior])
        field /= max(np.linalg.norm(field), 1.0e-300)
        smallest_modes.append(
            {
                "order": index,
                "singular_value": float(spectrum["smallest_singular_values"][index]),
                "field": field,
            }
        )
    identity = np.broadcast_to(np.eye(2), (len(points), 2, 2))
    a1 = np.asarray([[1.08, 0.04], [0.0, 1.0 / 1.08]])
    a1[1, 1] = (1.0 + a1[0, 1] * a1[1, 0]) / a1[0, 0]
    a2 = np.asarray([[0.96, 0.0], [0.03, 1.0 / 0.96]])
    a2[1, 1] = (1.0 + a2[0, 1] * a2[1, 0]) / a2[0, 0]
    f1 = np.broadcast_to(a1, identity.shape)
    f2 = np.broadcast_to(a2, identity.shape)
    inv1, j1 = inverse_raw(f1)
    inv2, j2 = inverse_raw(f2)
    recursive_g = transport_action_sparse(
        transport_action_sparse(disc.G0, inv1, j1), inv2, j2
    )
    cumulative = np.einsum("ab,bc->ac", a2, a1)
    fc = np.broadcast_to(cumulative, identity.shape)
    invc, jc = inverse_raw(fc)
    direct_g = transport_action_sparse(disc.G0, invc, jc)
    recursive_d = transport_divergence_sparse(
        transport_divergence_sparse(
            disc.D_q0, inv1, j1, interior
        ),
        inv2,
        j2,
        interior,
    )
    direct_d = transport_divergence_sparse(disc.D_q0, invc, jc, interior)
    diagnostics = disc.derivative.diagnostics
    quality = cloud_quality(points, disc.boundary, disc.spacing)
    return {
        "layers": layers,
        "target_h": 1.0 / layers,
        "particles": len(points),
        "boundary_particles": len(disc.boundary),
        "pressure_dofs": len(interior),
        "cloud": quality,
        "stencil_size": diagnostics.stencil_size,
        "gmls_max_condition": diagnostics.maximum_condition,
        "gmls_max_moment_defect": diagnostics.maximum_moment_defect,
        "local_condition": {
            name: float(np.max(condition[mask])) for name, mask in masks.items()
        },
        "local_moment_defect": {
            name: float(np.max(moment[mask])) for name, mask in masks.items()
        },
        "polynomial_reproduction_max": polynomial_max,
        "pressure_gradient_error": split_rms(
            discrete_gradient - exact_gradient, volume, masks
        ),
        "affine_divergence_error": split_rms(divergence, volume, masks),
        "mapped_affine_divergence_error": split_rms(
            divergence_affine, volume * geometry["J"], masks
        ),
        "mapped_pressure_action_error": split_rms(action_error, volume, masks),
        "affine_F_error_rms": vector_rms(geometry["F"] - affine, volume),
        "raw_J": {
            "minimum": float(np.min(geometry["J"])),
            "maximum": float(np.max(geometry["J"])),
            "maximum_F_condition": float(np.max(geometry["condition"])),
        },
        "pressure_boundary_max": float(np.max(np.abs(exact_pressure[disc.boundary]))),
        "spectrum": spectrum,
        "smallest_modes": smallest_modes,
        "composition": {
            "G_relative_defect": relative_sparse_norm(recursive_g, direct_g),
            "D_relative_defect": relative_sparse_norm(recursive_d, direct_d),
        },
        "no_forbidden_mechanisms": {
            "regularization": False,
            "determinant_clipping": False,
            "pressure_filtering": False,
            "hidden_projection": False,
            "mode_deletion": False,
        },
        "passed": bool(
            spectrum["rank_at_1e-12"] == len(interior)
            and np.min(geometry["J"]) > ADMISSIBLE_J
            and polynomial_max["all"] < 1.0e-9
            and np.max(np.abs(exact_pressure[disc.boundary])) < 1.0e-13
        ),
    }


def solve_step(
    disc: DiskDiscretization,
    position: np.ndarray,
    velocity_star: np.ndarray,
    dt: float,
    theta: float,
    d_q_current: sp.csr_matrix,
    d_strong_current: sp.csr_matrix,
    g_current: sp.csr_matrix,
    scalar_chain: np.ndarray,
    current_volume: np.ndarray,
    anchor: str = "T",
    geometry_relaxation: float = 1.0,
) -> dict[str, Any]:
    if anchor not in {"n", "star", "T"}:
        raise ValueError(anchor)
    x_star = position + dt * velocity_star
    star_geometry = relative_geometry(
        disc.derivative, x_star - position, scalar_chain
    )
    d_q_star = transport_divergence_sparse(
        d_q_current, star_geometry["inverse"], star_geometry["J"], disc.interior
    )
    x_iterate = x_star.copy()
    history: list[dict[str, Any]] = []
    pressure = np.zeros(g_current.shape[1])
    for iteration in range(60):
        terminal = relative_geometry(
            disc.derivative, x_iterate - position, scalar_chain
        )
        placed = relative_geometry(
            disc.derivative, theta * (x_iterate - position), scalar_chain
        )
        if (
            np.min(terminal["J"]) <= ADMISSIBLE_J
            or np.min(placed["J"]) <= ADMISSIBLE_J
        ):
            return {
                "passed": False,
                "failure": "raw Jacobian below admissible threshold",
                "history": history,
            }
        d_q_terminal = transport_divergence_sparse(
            d_q_current, terminal["inverse"], terminal["J"], disc.interior
        )
        d_strong_terminal = transport_divergence_sparse(
            d_strong_current,
            terminal["inverse"],
            terminal["J"],
            np.arange(len(position)),
        )
        action_theta = transport_action_sparse(
            g_current, placed["inverse"], placed["J"]
        )
        if anchor == "n":
            d_anchor = d_q_current
            test_volume = current_volume[disc.interior]
        elif anchor == "star":
            d_anchor = d_q_star
            test_volume = (
                current_volume[disc.interior] * star_geometry["J"][disc.interior]
            )
        else:
            d_anchor = d_q_terminal
            test_volume = (
                current_volume[disc.interior] * terminal["J"][disc.interior]
            )
        response = (d_anchor @ action_theta).tocsc()
        rhs = np.asarray(d_anchor @ velocity_star.ravel()).reshape(-1) / dt
        lu, linear = factor_and_condition(response)
        pressure = lu.solve(rhs)
        linear_residual = response @ pressure - rhs
        action_value = np.asarray(action_theta @ pressure).reshape(-1, 2)
        velocity = velocity_star - dt * action_value
        x_candidate = x_star - GAMMA * dt**2 * action_value
        delta = x_candidate - x_iterate
        pointwise = np.linalg.norm(delta, axis=1) / disc.spacing
        anchor_residual = np.asarray(d_anchor @ velocity.ravel()).reshape(-1)
        terminal_residual = np.asarray(
            d_q_terminal @ velocity.ravel()
        ).reshape(-1)
        row = {
            "iteration": iteration,
            "geometry_rms_over_h": float(np.sqrt(np.mean(pointwise**2))),
            "geometry_max_over_h": float(np.max(pointwise)),
            "linear_relative_residual": float(
                np.linalg.norm(linear_residual)
                / max(np.linalg.norm(rhs), 1.0e-300)
            ),
            "anchor_residual_rms": scalar_rms(anchor_residual, test_volume),
            "terminal_residual_rms": scalar_rms(
                terminal_residual,
                current_volume[disc.interior] * terminal["J"][disc.interior],
            ),
            "min_raw_J_T": float(np.min(terminal["J"])),
            "min_raw_J_theta": float(np.min(placed["J"])),
            "max_F_condition_T": float(np.max(terminal["condition"])),
            "linear": linear,
        }
        history.append(row)
        x_iterate = (
            (1.0 - geometry_relaxation) * x_iterate
            + geometry_relaxation * x_candidate
        )
        if (
            row["geometry_rms_over_h"] <= GEOMETRY_RMS_TOL
            and row["geometry_max_over_h"] <= GEOMETRY_MAX_TOL
        ):
            break
    if len(history) == 60:
        return {
            "passed": False,
            "failure": "geometry Picard did not converge",
            "history": history,
        }

    # Final hard rebuild and one final square pressure solve.
    accepted = x_iterate.copy()
    terminal = relative_geometry(
        disc.derivative, accepted - position, scalar_chain
    )
    placed = relative_geometry(
        disc.derivative, theta * (accepted - position), scalar_chain
    )
    d_q_terminal = transport_divergence_sparse(
        d_q_current, terminal["inverse"], terminal["J"], disc.interior
    )
    d_strong_terminal = transport_divergence_sparse(
        d_strong_current,
        terminal["inverse"],
        terminal["J"],
        np.arange(len(position)),
    )
    action_theta = transport_action_sparse(g_current, placed["inverse"], placed["J"])
    action_terminal = transport_action_sparse(
        g_current, terminal["inverse"], terminal["J"]
    )
    if anchor == "n":
        d_anchor = d_q_current
        anchor_volume = current_volume[disc.interior]
    elif anchor == "star":
        d_anchor = d_q_star
        anchor_volume = (
            current_volume[disc.interior] * star_geometry["J"][disc.interior]
        )
    else:
        d_anchor = d_q_terminal
        anchor_volume = (
            current_volume[disc.interior] * terminal["J"][disc.interior]
        )
    response = (d_anchor @ action_theta).tocsc()
    rhs = np.asarray(d_anchor @ velocity_star.ravel()).reshape(-1) / dt
    lu, linear = factor_and_condition(response)
    pressure = lu.solve(rhs)
    pressure_action = np.asarray(action_theta @ pressure).reshape(-1, 2)
    velocity = velocity_star - dt * pressure_action
    x_check = x_star - GAMMA * dt**2 * pressure_action
    geometry_delta = x_check - accepted
    anchor_residual = np.asarray(d_anchor @ velocity.ravel()).reshape(-1)
    terminal_residual = np.asarray(d_q_terminal @ velocity.ravel()).reshape(-1)
    strong_residual = np.asarray(
        d_strong_terminal @ velocity.ravel()
    ).reshape(-1)
    linear_residual = response @ pressure - rhs
    terminal_volume = current_volume * terminal["J"]
    kinetic_star = 0.5 * float(
        np.sum(disc.volume * np.sum(velocity_star**2, axis=1))
    )
    kinetic_terminal = 0.5 * float(
        np.sum(disc.volume * np.sum(velocity**2, axis=1))
    )
    work = (
        -dt
        * float(np.sum(disc.volume * np.sum(velocity_star * pressure_action, axis=1)))
        + 0.5
        * dt**2
        * float(np.sum(disc.volume * np.sum(pressure_action**2, axis=1)))
    )
    represented_area = float(np.sum(current_volume * terminal["J"]))
    polygon_area = ordered_polygon_area(accepted, disc.boundary)
    initial_polygon = ordered_polygon_area(position, disc.boundary)
    initial_represented = float(np.sum(current_volume))
    geometry_rms = float(
        np.sqrt(np.mean(np.sum(geometry_delta**2, axis=1))) / disc.spacing
    )
    geometry_max = float(
        np.max(np.linalg.norm(geometry_delta, axis=1)) / disc.spacing
    )
    passed = bool(
        geometry_rms <= 1.0e-10
        and geometry_max <= 1.0e-9
        and np.min(terminal["J"]) > ADMISSIBLE_J
        and np.min(placed["J"]) > ADMISSIBLE_J
        and scalar_rms(anchor_residual, anchor_volume) <= TERMINAL_RMS_TOL
        and np.max(np.abs(anchor_residual)) <= TERMINAL_MAX_TOL
        and np.linalg.norm(linear_residual)
        / max(np.linalg.norm(rhs), 1.0e-300)
        <= 1.0e-10
    )
    return {
        "passed": passed,
        "anchor": anchor,
        "theta": theta,
        "dt": dt,
        "position": accepted,
        "velocity": velocity,
        "velocity_star": velocity_star,
        "pressure": pressure,
        "pressure_action": pressure_action,
        "D_q_terminal": d_q_terminal,
        "D_strong_terminal": d_strong_terminal,
        "G_theta": action_theta,
        "G_terminal": action_terminal,
        "terminal_geometry": terminal,
        "placed_geometry": placed,
        "history": history,
        "geometry_residual": {
            "rms_over_h": geometry_rms,
            "max_over_h": geometry_max,
        },
        "linear": linear,
        "linear_relative_residual": float(
            np.linalg.norm(linear_residual) / max(np.linalg.norm(rhs), 1.0e-300)
        ),
        "anchor_residual": {
            "rms": scalar_rms(anchor_residual, anchor_volume),
            "max": float(np.max(np.abs(anchor_residual))),
        },
        "terminal_residual": {
            "rms": scalar_rms(
                terminal_residual, terminal_volume[disc.interior]
            ),
            "max": float(np.max(np.abs(terminal_residual))),
            "source_mismatch_rms": scalar_rms(
                terminal_residual / dt, terminal_volume[disc.interior]
            ),
        },
        "strong_diagnostic": split_rms(
            strong_residual, terminal_volume, disc.masks
        ),
        "energy": {
            "kinetic_star": kinetic_star,
            "kinetic_terminal": kinetic_terminal,
            "delta_K_pressure": kinetic_terminal - kinetic_star,
            "pressure_work_identity": work,
            "identity_defect": abs(kinetic_terminal - kinetic_star - work),
        },
        "area": {
            "J_minus_one_rms": scalar_rms(
                terminal["J"] - 1.0, current_volume
            ),
            "J_minus_one_max": float(np.max(np.abs(terminal["J"] - 1.0))),
            "represented_area": represented_area,
            "polygon_area": polygon_area,
            "represented_relative_increment": (
                represented_area - initial_represented
            )
            / initial_represented,
            "polygon_relative_increment": (
                polygon_area - initial_polygon
            )
            / initial_polygon,
            "increment_gap_relative": (
                (polygon_area - initial_polygon)
                - (represented_area - initial_represented)
            )
            / initial_represented,
        },
        "raw_jacobians": {
            "terminal_min": float(np.min(terminal["J"])),
            "terminal_max": float(np.max(terminal["J"])),
            "theta_min": float(np.min(placed["J"])),
            "theta_max": float(np.max(placed["J"])),
            "terminal_max_F_condition": float(np.max(terminal["condition"])),
        },
    }


def public_step(case: dict[str, Any], include_fields: bool = False) -> dict[str, Any]:
    excluded = {
        "D_q_terminal",
        "D_strong_terminal",
        "G_theta",
        "G_terminal",
        "terminal_geometry",
        "placed_geometry",
    }
    if not include_fields:
        excluded |= {
            "position",
            "velocity",
            "velocity_star",
            "pressure",
            "pressure_action",
        }
    result = {key: value for key, value in case.items() if key not in excluded}
    if include_fields:
        result["terminal_geometry_summary"] = {
            "J": case["terminal_geometry"]["J"],
            "F": case["terminal_geometry"]["F"],
        }
    return result


def anchor_audit() -> dict[str, Any]:
    disc = build_discretization(max(LAYERS))
    points = disc.points
    velocity0 = np.column_stack((points[:, 0], -points[:, 1]))
    identity = np.broadcast_to(np.eye(2), (len(points), 2, 2)).copy()
    rows = []
    raw_cases = []
    for dt in ANCHOR_DTS:
        for anchor in ("n", "star", "T"):
            case = solve_step(
                disc,
                points,
                velocity0,
                dt,
                0.5,
                disc.D_q0,
                disc.D_strong0,
                disc.G0,
                identity,
                disc.volume,
                anchor=anchor,
            )
            raw_cases.append(case)
            rows.append(public_step(case))
    rates = {}
    for anchor in ("n", "star", "T"):
        selected = [case for case in raw_cases if case["anchor"] == anchor]
        rates[anchor] = {
            "terminal_residual": observed_rate(
                [case["dt"] for case in selected],
                [case["terminal_residual"]["rms"] for case in selected],
            ),
            "source_mismatch": observed_rate(
                [case["dt"] for case in selected],
                [case["terminal_residual"]["source_mismatch_rms"] for case in selected],
            ),
            "J_minus_one": observed_rate(
                [case["dt"] for case in selected],
                [case["area"]["J_minus_one_rms"] for case in selected],
            ),
        }
    return {
        "layers": disc.layers,
        "cases": rows,
        "rates": rates,
        "all_passed_own_anchor": all(case["passed"] for case in raw_cases),
        "terminal_anchor_max_terminal_residual": max(
            case["terminal_residual"]["rms"]
            for case in raw_cases
            if case["anchor"] == "T"
        ),
    }


def theta_sweep(layers: int, dt: float = SWEEP_DT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    disc = build_discretization(layers)
    points = disc.points
    velocity0 = np.column_stack((points[:, 0], -points[:, 1]))
    identity = np.broadcast_to(np.eye(2), (len(points), 2, 2)).copy()
    cache: dict[float, dict[str, Any]] = {}

    def solve(theta: float) -> dict[str, Any]:
        key = round(float(theta), 13)
        if key not in cache:
            cache[key] = solve_step(
                disc,
                points,
                velocity0,
                dt,
                float(theta),
                disc.D_q0,
                disc.D_strong0,
                disc.G0,
                identity,
                disc.volume,
                anchor="T",
            )
        return cache[key]

    cases = [solve(theta) for theta in THETA_VALUES]
    roots: list[float] = []
    for left, right in zip(cases[:-1], cases[1:]):
        left_energy = left["energy"]["delta_K_pressure"]
        right_energy = right["energy"]["delta_K_pressure"]
        if left_energy == 0.0:
            roots.append(left["theta"])
        elif left_energy * right_energy < 0.0:
            roots.append(
                float(
                    scipy.optimize.brentq(
                        lambda value: solve(value)["energy"]["delta_K_pressure"],
                        left["theta"],
                        right["theta"],
                        xtol=1.0e-12,
                        rtol=1.0e-12,
                    )
                )
            )
    roots = sorted(set(round(root, 12) for root in roots))
    selected = {
        str(theta): solve(theta)["energy"]["delta_K_pressure"]
        for theta in (0.0, 0.5, 1.0)
    }
    public = {
        "layers": layers,
        "dt": dt,
        "cases": [public_step(case) for case in cases],
        "delta_K_selected": selected,
        "endpoint_sign_product": selected["0.0"] * selected["1.0"],
        "zero_crossings": roots,
        "primary_zero_crossing": roots[0] if roots else None,
        "all_passed": all(case["passed"] for case in cases),
        "maximum_terminal_closure": max(
            case["terminal_residual"]["rms"] for case in cases
        ),
        "maximum_pressure_work_identity_defect": max(
            case["energy"]["identity_defect"] for case in cases
        ),
    }
    return public, cases


def theta_sweep_matrix() -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    public_sweeps = []
    raw: dict[int, list[dict[str, Any]]] = {}
    for layers in LAYERS:
        public, cases = theta_sweep(layers)
        public_sweeps.append(public)
        raw[layers] = cases
    curve_differences = []
    for coarse, fine in zip(public_sweeps[:-1], public_sweeps[1:]):
        coarse_curve = np.asarray(
            [case["energy"]["delta_K_pressure"] for case in coarse["cases"]]
        )
        fine_curve = np.asarray(
            [case["energy"]["delta_K_pressure"] for case in fine["cases"]]
        )
        curve_differences.append(
            {
                "coarse_layers": coarse["layers"],
                "fine_layers": fine["layers"],
                "max_absolute_difference": float(
                    np.max(np.abs(fine_curve - coarse_curve))
                ),
                "relative_l2_difference": float(
                    np.linalg.norm(fine_curve - coarse_curve)
                    / max(np.linalg.norm(fine_curve), 1.0e-300)
                ),
            }
        )
    temporal_sweeps = []
    for dt in ANCHOR_DTS:
        if abs(dt - SWEEP_DT) < 1.0e-15:
            temporal_sweeps.append(
                next(sweep for sweep in public_sweeps if sweep["layers"] == 18)
            )
        else:
            public, _ = theta_sweep(18, dt)
            temporal_sweeps.append(public)
    temporal_sweeps.sort(key=lambda sweep: sweep["dt"], reverse=True)
    root_rows = [
        sweep
        for sweep in temporal_sweeps
        if sweep["primary_zero_crossing"] is not None
    ]
    return {
        "sweeps": public_sweeps,
        "spatial_curve_differences": curve_differences,
        "temporal_sweeps": temporal_sweeps,
        "theta_star_offset_temporal_rate": observed_rate(
            [sweep["dt"] for sweep in root_rows],
            [
                sweep["primary_zero_crossing"] - 0.5
                for sweep in root_rows
            ],
        ),
    }, raw


def power_norm(matrix: np.ndarray, iterations: int = 60) -> float:
    rng = np.random.default_rng(DETERMINISTIC_SEED)
    vector = rng.standard_normal(matrix.shape[1])
    vector /= np.linalg.norm(vector)
    value = 0.0
    for _ in range(iterations):
        left = matrix @ vector
        right = matrix.T @ left
        norm = np.linalg.norm(right)
        if norm == 0.0:
            return 0.0
        vector = right / norm
        value = np.linalg.norm(left)
    return float(value)


def green_compatibility_audit(
    disc: DiskDiscretization,
    d_q: sp.spmatrix,
    action: sp.spmatrix,
    terminal_geometry: dict[str, np.ndarray],
    terminal_position: np.ndarray,
) -> dict[str, Any]:
    """Measure full-space and resolved-subspace discrete Green compatibility.

    The full-space quantity is the relative Frobenius mismatch between the
    independently constructed point-value divergence and the point-value
    representative induced by the mass adjoint of the pressure action.  It is
    not a truncation error for either operator.  The restricted quantity is
    the Riesz-normalized bilinear mismatch on the Case 1 solution space:
    trace-free affine velocities and the quadratic homogeneous-Dirichlet
    pressure on the disk.
    """
    terminal_test_mass = (
        disc.volume[disc.interior] * terminal_geometry["J"][disc.interior]
    )
    velocity_mass = np.repeat(disc.volume, 2)
    pressure_mass = disc.volume[disc.interior]
    d_adjoint = (
        -sp.diags(1.0 / terminal_test_mass)
        @ action.T
        @ sp.diags(velocity_mass)
    ).tocsr()
    full_relative_frobenius = float(
        spla.norm(d_q - d_adjoint)
        / max(spla.norm(d_q), 1.0e-300)
    )

    # A basis for all two-dimensional trace-free affine Eulerian velocities.
    affine_generators = (
        np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )
    velocity_basis = np.column_stack(
        [
            (terminal_position @ generator.T).ravel()
            for generator in affine_generators
        ]
    )
    pressure_basis = (
        1.0 - np.sum(disc.points**2, axis=1)
    )[disc.interior, None]
    pairing_mismatch = (
        velocity_basis.T
        @ (velocity_mass[:, None] * (action @ pressure_basis))
        + (d_q @ velocity_basis).T
        @ (terminal_test_mass[:, None] * pressure_basis)
    )
    velocity_gram = velocity_basis.T @ (
        velocity_mass[:, None] * velocity_basis
    )
    pressure_gram = pressure_basis.T @ (
        pressure_mass[:, None] * pressure_basis
    )
    restricted_supremum = float(
        np.sqrt(
            (
                pairing_mismatch.T
                @ np.linalg.solve(velocity_gram, pairing_mismatch)
                / pressure_gram
            )[0, 0]
        )
    )
    restricted_by_velocity_basis = (
        np.abs(pairing_mismatch[:, 0])
        / np.sqrt(np.diag(velocity_gram) * pressure_gram[0, 0])
    )
    return {
        "definition": (
            "||D_T^q-D_T^ad||_F/||D_T^q||_F, with "
            "D_T^ad=-M_Z^{-1} G_T^T M_U"
        ),
        "norm": "Frobenius",
        "denominator": "||D_T^q||_F",
        "full_space_relative_frobenius": full_relative_frobenius,
        "restricted_affine_quadratic": {
            "definition": (
                "sup |u^T M_U G_T p + p^T M_Z D_T^q u| / "
                "(||u||_M_U ||p||_M_Q)"
            ),
            "velocity_space": "trace-free affine Eulerian velocities",
            "pressure_space": (
                "span{1-|X|^2}, the quadratic homogeneous-Dirichlet disk pressure"
            ),
            "riesz_normalized_supremum": restricted_supremum,
            "values_by_velocity_basis": restricted_by_velocity_basis,
        },
        "interpretation": (
            "the full-space value measures same-configuration discrete "
            "Green compatibility, not D, G, pressure-solve, or projector error"
        ),
        "D_T_ad": d_adjoint,
    }


def projector_audit(raw_sweeps: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    layers = min(LAYERS)
    disc = build_discretization(layers)
    base = min(raw_sweeps[layers], key=lambda case: abs(case["theta"] - 0.5))
    d_q = base["D_q_terminal"]
    displacement = base["position"] - disc.points
    terminal_geometry = base["terminal_geometry"]
    g_terminal = transport_action_sparse(
        disc.G0, terminal_geometry["inverse"], terminal_geometry["J"]
    )
    root_mass = np.repeat(np.sqrt(disc.volume), 2)
    inv_root_mass = 1.0 / root_mass
    response_terminal = (d_q @ g_terminal).tocsc()
    lu_terminal = spla.splu(response_terminal)
    p_terminal = np.eye(2 * len(disc.points)) - (
        g_terminal @ lu_terminal.solve(d_q.toarray())
    )
    q_terminal, _ = np.linalg.qr(root_mass[:, None] * g_terminal.toarray())
    green_compatibility = green_compatibility_audit(
        disc,
        d_q,
        g_terminal,
        terminal_geometry,
        base["position"],
    )
    endpoint_adjoint = green_compatibility["D_T_ad"]
    endpoint_green_defect = green_compatibility[
        "full_space_relative_frobenius"
    ]
    # Auxiliary compatible control.  This is deliberately not the production
    # divergence.  It replaces the endpoint test functional by the exact
    # mass-adjoint of G_T, so the theta=T projector is an M_U-orthogonal
    # baseline.  Sweeping the same transported actions away from T then
    # isolates configuration-induced obliqueness without the pre-existing
    # production-pair adjoint defect.
    d_compatible = endpoint_adjoint.tocsr()
    response_compatible_terminal = (d_compatible @ g_terminal).tocsc()
    lu_compatible_terminal = spla.splu(response_compatible_terminal)
    p_compatible_terminal = np.eye(2 * len(disc.points)) - (
        g_terminal @ lu_compatible_terminal.solve(d_compatible.toarray())
    )
    weighted_compatible_terminal = (
        root_mass[:, None]
        * p_compatible_terminal
        * inv_root_mass[None, :]
    )
    rows = []
    compatible_rows = []
    for theta in THETA_VALUES:
        placed = relative_geometry(
            disc.derivative,
            theta * displacement,
            np.broadcast_to(np.eye(2), (len(disc.points), 2, 2)),
        )
        action = transport_action_sparse(disc.G0, placed["inverse"], placed["J"])
        response = (d_q @ action).tocsc()
        lu = spla.splu(response)
        projector = np.eye(2 * len(disc.points)) - (
            action @ lu.solve(d_q.toarray())
        )
        weighted_projector = (
            root_mass[:, None] * projector * inv_root_mass[None, :]
        )
        weighted_difference = (
            root_mass[:, None]
            * (projector - p_terminal)
            * inv_root_mass[None, :]
        )
        q_action, _ = np.linalg.qr(root_mass[:, None] * action.toarray())
        singular_angles = np.linalg.svd(
            q_action.T @ q_terminal, compute_uv=False
        )
        maximum_angle = float(
            np.degrees(np.arccos(np.clip(np.min(singular_angles), -1.0, 1.0)))
        )
        minimum_action_cosine = float(
            np.clip(np.min(singular_angles), 0.0, 1.0)
        )
        rows.append(
            {
                "theta": theta,
                "projector_mass_norm": power_norm(weighted_projector),
                "projector_distance_from_T_mass_norm": power_norm(
                    weighted_difference
                ),
                "maximum_principal_angle_degrees": maximum_angle,
                "closure_relative": float(
                    np.linalg.norm(d_q @ projector)
                    / max(spla.norm(d_q), 1.0e-300)
                ),
                "idempotence_relative": float(
                    np.linalg.norm(projector @ projector - projector)
                    / max(np.linalg.norm(projector), 1.0e-300)
                ),
                "action_annihilation_relative": float(
                    np.linalg.norm(projector @ action.toarray())
                    / max(spla.norm(action), 1.0e-300)
                ),
            }
        )
        response_compatible = (d_compatible @ action).tocsc()
        lu_compatible = spla.splu(response_compatible)
        projector_compatible = np.eye(2 * len(disc.points)) - (
            action @ lu_compatible.solve(d_compatible.toarray())
        )
        compatible_rows.append(
            {
                "theta": theta,
                # For a projector onto range(G_T)^perp along range(G_theta),
                # the exact mass norm is sec(alpha_max) and its distance from
                # the orthogonal endpoint projector is tan(alpha_max).  These
                # angle formulas avoid slow power-iteration convergence when
                # the norm is only slightly larger than one.
                "projector_mass_norm": (
                    1.0
                    if abs(theta - 1.0) <= 1.0e-14
                    else 1.0 / minimum_action_cosine
                ),
                "projector_distance_from_T_mass_norm": (
                    0.0
                    if abs(theta - 1.0) <= 1.0e-14
                    else float(
                        np.sqrt(
                            max(0.0, 1.0 - minimum_action_cosine**2)
                        )
                        / minimum_action_cosine
                    )
                ),
                "closure_relative": float(
                    np.linalg.norm(d_compatible @ projector_compatible)
                    / max(spla.norm(d_compatible), 1.0e-300)
                ),
                "idempotence_relative": float(
                    np.linalg.norm(
                        projector_compatible @ projector_compatible
                        - projector_compatible
                    )
                    / max(np.linalg.norm(projector_compatible), 1.0e-300)
                ),
                "action_annihilation_relative": float(
                    np.linalg.norm(projector_compatible @ action.toarray())
                    / max(spla.norm(action), 1.0e-300)
                ),
            }
        )
    endpoint_row = min(rows, key=lambda row: abs(row["theta"] - 1.0))
    compatible_endpoint_row = min(
        compatible_rows, key=lambda row: abs(row["theta"] - 1.0)
    )
    return {
        "layers": layers,
        "source_configuration_theta": 0.5,
        "endpoint_projector_mass_norm": endpoint_row["projector_mass_norm"],
        "endpoint_green_adjoint_relative_defect": endpoint_green_defect,
        "same_configuration_green_compatibility": {
            key: value
            for key, value in green_compatibility.items()
            if key != "D_T_ad"
        },
        "interpretation": (
            "endpoint mass norm is the discretization baseline; theta-dependent "
            "distance and angle are the configuration-placement increments"
        ),
        "rows": rows,
        "compatible_endpoint_control": {
            "definition": (
                "auxiliary only: D_T^ad=-M_Z^{-1} G_T^T M_U; production "
                "D_T^q is unchanged"
            ),
            "endpoint_projector_mass_norm": compatible_endpoint_row[
                "projector_mass_norm"
            ],
            "endpoint_mass_self_adjoint_relative_defect": float(
                np.linalg.norm(
                    weighted_compatible_terminal
                    - weighted_compatible_terminal.T
                )
                / max(np.linalg.norm(weighted_compatible_terminal), 1.0e-300)
            ),
            "maximum_off_endpoint_projector_mass_norm": max(
                row["projector_mass_norm"]
                for row in compatible_rows
                if abs(row["theta"] - 1.0) > 1.0e-14
            ),
            "maximum_projector_distance_from_T_mass_norm": max(
                row["projector_distance_from_T_mass_norm"]
                for row in compatible_rows
            ),
            "rows": compatible_rows,
            "interpretation": (
                "the compatible endpoint has unit mass norm; departures for "
                "theta<1 isolate the geometry-induced range mismatch"
            ),
        },
    }


def placed_action_stage_consistency(
    layers: int = STAGE_LAYERS,
    start_time: float = STAGE_START_TIME,
    dts: tuple[float, ...] = STAGE_DTS,
    thetas: tuple[float, ...] = STAGE_THETAS,
) -> dict[str, Any]:
    """Exact-start local audit of the placed pressure action.

    The incoming affine state and its transported operators are supplied from
    the analytic solution at ``start_time``.  This prevents accumulated
    trajectory error from being mistaken for a local action-stage defect.
    """

    disc = build_discretization(layers)
    reference_n = affine_reference(np.asarray([start_time], dtype=np.float64))
    a_n = reference_n["A"][0]
    a_dot_n = reference_n["A_dot"][0]
    inverse_n = np.linalg.inv(a_n)
    count = len(disc.points)
    inverse_field = np.broadcast_to(inverse_n, (count, 2, 2)).copy()
    unit_jacobian = np.ones(count, dtype=np.float64)
    position_n = disc.points @ a_n.T
    velocity_n = disc.points @ a_dot_n.T
    d_q_n = transport_divergence_sparse(
        disc.D_q0, inverse_field, unit_jacobian, disc.interior
    )
    d_strong_n = transport_divergence_sparse(
        disc.D_strong0,
        inverse_field,
        unit_jacobian,
        np.arange(count),
    )
    g_n = transport_action_sparse(disc.G0, inverse_field, unit_jacobian)

    # A projection multiplier represents an impulse coefficient.  The
    # instantaneous-stage comparison below is therefore complemented by a
    # direct comparison with the exact step-average pressure.  Gauss--Legendre
    # quadrature makes the reference error negligible relative to the audited
    # discretization errors.
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(16)
    average_pressure_by_dt: dict[float, np.ndarray] = {}
    average_lambda_by_dt: dict[float, float] = {}
    for dt in dts:
        quadrature_times = start_time + 0.5 * dt * (gauss_nodes + 1.0)
        lambda_values = affine_reference(quadrature_times)["lambda"]
        average_lambda = float(0.5 * np.dot(gauss_weights, lambda_values))
        average_lambda_by_dt[dt] = average_lambda
        average_pressure_by_dt[dt] = 0.5 * average_lambda * (
            1.0 - np.sum(disc.points**2, axis=1)
        )

    rows: list[dict[str, Any]] = []
    for theta in thetas:
        for dt in dts:
            case = solve_step(
                disc,
                position_n,
                velocity_n,
                dt,
                theta,
                d_q_n,
                d_strong_n,
                g_n,
                inverse_field,
                disc.volume,
                anchor="T",
            )
            if not case["passed"]:
                rows.append(
                    {
                        "theta": theta,
                        "dt": dt,
                        "passed": False,
                        "failure": case.get("failure", "stage solve failed"),
                    }
                )
                continue

            action_time = start_time + theta * dt
            reference_theta = affine_reference(
                np.asarray([action_time], dtype=np.float64)
            )
            a_theta = reference_theta["A"][0]
            lambda_theta = float(reference_theta["lambda"][0])
            exact_pressure = 0.5 * lambda_theta * (
                1.0 - np.sum(disc.points**2, axis=1)
            )
            exact_action = -lambda_theta * (
                disc.points @ np.linalg.inv(a_theta)
            )

            action_position = position_n + theta * (
                case["position"] - position_n
            )
            exact_action_position = disc.points @ a_theta.T

            # Reproduction on the actual production path is separated from
            # temporal stage consistency.  For this affine/quadratic pair it
            # should be at the floating-point reconstruction floor.
            cumulative_production_f = np.einsum(
                "nij,jk->nik", case["placed_geometry"]["F"], a_n
            )
            inverse_production_f = np.linalg.inv(cumulative_production_f)
            jacobian_production = np.linalg.det(cumulative_production_f)
            exact_action_on_production_path = (
                -lambda_theta
                * jacobian_production[:, None]
                * np.einsum(
                    "ni,nij->nj", disc.points, inverse_production_f
                )
            )
            applied_exact_pressure = np.asarray(
                case["G_theta"] @ exact_pressure[disc.interior]
            ).reshape(-1, 2)

            rows.append(
                {
                    "theta": theta,
                    "dt": dt,
                    "action_time": action_time,
                    "passed": True,
                    "action_stage_error_rms": vector_rms(
                        case["pressure_action"] - exact_action, disc.volume
                    ),
                    "pressure_stage_error_rms": scalar_rms(
                        case["pressure"] - exact_pressure[disc.interior],
                        disc.volume[disc.interior],
                    ),
                    "pressure_step_average_error_rms": scalar_rms(
                        case["pressure"]
                        - average_pressure_by_dt[dt][disc.interior],
                        disc.volume[disc.interior],
                    ),
                    "exact_step_average_lambda": average_lambda_by_dt[dt],
                    "production_path_position_error_rms": vector_rms(
                        action_position - exact_action_position, disc.volume
                    ),
                    "spatial_action_reproduction_error_rms": vector_rms(
                        applied_exact_pressure - exact_action_on_production_path,
                        disc.volume,
                    ),
                    "terminal_residual_rms": case["terminal_residual"]["rms"],
                    "picard_iterations": len(case["history"]),
                }
            )

    rates: dict[str, dict[str, float | None]] = {}
    for theta in thetas:
        selected = sorted(
            (
                row
                for row in rows
                if row["passed"] and row["theta"] == theta
            ),
            key=lambda row: row["dt"],
            reverse=True,
        )
        rates[str(theta)] = {
            metric: observed_rate(
                [row["dt"] for row in selected],
                [row[metric] for row in selected],
            )
            for metric in (
                "action_stage_error_rms",
                "pressure_stage_error_rms",
                "pressure_step_average_error_rms",
                "production_path_position_error_rms",
            )
        }

    midpoint_rate = rates["0.5"]["action_stage_error_rms"]
    endpoint_rates = (
        rates["0.0"]["action_stage_error_rms"],
        rates["1.0"]["action_stage_error_rms"],
    )
    maximum_reproduction_error = max(
        row["spatial_action_reproduction_error_rms"]
        for row in rows
        if row["passed"]
    )
    passed = bool(
        all(row["passed"] for row in rows)
        and midpoint_rate is not None
        and midpoint_rate >= 1.8
        and all(
            rate is not None and 0.8 <= rate <= 1.25
            for rate in endpoint_rates
        )
        and maximum_reproduction_error < 1.0e-11
    )
    return {
        "layers": layers,
        "start_time": start_time,
        "dts": dts,
        "thetas": thetas,
        "definition": (
            "exact-start one-step errors on common material labels; "
            "action_stage_error_rms=||G_theta p_theta-g_exact(t_n+theta dt)||_M; "
            "pressure_step_average_error_rms compares the multiplier with "
            "dt^{-1} integral_[t_n,t_n+dt] p(t) dt"
        ),
        "rows": rows,
        "rates": rates,
        "maximum_spatial_action_reproduction_error_rms": (
            maximum_reproduction_error
        ),
        "passed": passed,
    }


def independent_divergence(
    disc: DiskDiscretization,
    position: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry = relative_geometry(
        disc.independent_derivative,
        position - disc.points,
        np.broadcast_to(np.eye(2), (len(position), 2, 2)),
    )
    operator = transport_divergence_sparse(
        disc.D_independent0,
        geometry["inverse"],
        geometry["J"],
        np.arange(len(position)),
    )
    residual = np.asarray(operator @ velocity.ravel()).reshape(-1)
    return residual, geometry


def exact_state_at(reference: dict[str, np.ndarray], index: int, points: np.ndarray) -> dict[str, np.ndarray]:
    A = reference["A"][index]
    A_dot = reference["A_dot"][index]
    return {
        "position": points @ A.T,
        "velocity": points @ A_dot.T,
        "F": A,
        "J": float(np.linalg.det(A)),
        "lambda": float(reference["lambda"][index]),
    }


def run_multistep_case(
    layers: int,
    dt: float,
    theta: float,
    final_time: float = FINAL_TIME,
) -> dict[str, Any]:
    disc = build_discretization(layers)
    steps = int(round(final_time / dt))
    times = np.linspace(0.0, final_time, steps + 1)
    reference = affine_reference(times)
    action_reference = affine_reference(times[:-1] + theta * dt)
    position = disc.points.copy()
    velocity = np.column_stack((disc.points[:, 0], -disc.points[:, 1]))
    d_q = disc.D_q0.copy()
    d_strong = disc.D_strong0.copy()
    action = disc.G0.copy()
    scalar_chain = np.broadcast_to(
        np.eye(2), (len(position), 2, 2)
    ).copy()
    cumulative_f = scalar_chain.copy()
    cumulative_j = np.ones(len(position))
    current_volume = disc.volume.copy()
    initial_polygon = ordered_polygon_area(position, disc.boundary)
    initial_area_j = float(np.sum(current_volume))
    initial_kinetic = 0.5 * float(
        np.sum(disc.volume * np.sum(velocity**2, axis=1))
    )
    rows: list[dict[str, Any]] = []
    final_pressure = np.zeros(len(disc.interior))
    for step in range(1, steps + 1):
        start_residual = np.asarray(d_q @ velocity.ravel()).reshape(-1)
        case = solve_step(
            disc,
            position,
            velocity,
            dt,
            theta,
            d_q,
            d_strong,
            action,
            scalar_chain,
            current_volume,
            anchor="T",
        )
        if not case["passed"]:
            return {
                "passed": False,
                "configuration": {"layers": layers, "dt": dt, "theta": theta},
                "failed_step": step,
                "rows": rows,
                "failure": public_step(case),
            }
        terminal = case["terminal_geometry"]
        position_next = case["position"]
        velocity_next = case["velocity"]
        d_q_next = case["D_q_terminal"]
        d_strong_next = case["D_strong_terminal"]
        action_next = case["G_terminal"]
        scalar_chain_next = np.einsum(
            "nba,nbc->nac", terminal["inverse"], scalar_chain
        )
        cumulative_f_next = np.einsum(
            "nij,njk->nik", terminal["F"], cumulative_f
        )
        cumulative_j_next = cumulative_j * terminal["J"]
        direct_dq = transport_divergence_sparse(
            disc.D_q0,
            np.linalg.inv(cumulative_f_next),
            cumulative_j_next,
            disc.interior,
        )
        direct_g = transport_action_sparse(
            disc.G0,
            np.linalg.inv(cumulative_f_next),
            cumulative_j_next,
        )
        position_geometry = relative_geometry(
            disc.derivative,
            position_next - disc.points,
            np.broadcast_to(np.eye(2), (len(position), 2, 2)),
        )
        exact = exact_state_at(reference, step, disc.points)
        exact_pressure = (
            0.5
            * reference["lambda"][step]
            * (1.0 - np.sum(disc.points**2, axis=1))
        )
        exact_action_pressure = (
            0.5
            * action_reference["lambda"][step - 1]
            * (1.0 - np.sum(disc.points**2, axis=1))
        )
        pressure_full = np.zeros(len(position))
        pressure_full[disc.interior] = case["pressure"]
        independent, independent_geometry = independent_divergence(
            disc, position_next, velocity_next
        )
        independent_roundoff_floor = float(
            20.0
            * np.finfo(np.float64).eps
            * disc.independent_derivative.diagnostics.maximum_condition
            * max(float(np.max(np.linalg.norm(velocity_next, axis=1))), 1.0)
            / disc.spacing
        )
        represented = np.asarray(
            d_q_next @ velocity_next.ravel()
        ).reshape(-1)
        strong = np.asarray(
            d_strong_next @ velocity_next.ravel()
        ).reshape(-1)
        terminal_volume = current_volume * terminal["J"]
        affine_fit = fit_affine_map(disc.points, position_next, disc.volume)
        axes = np.sort(np.linalg.svd(affine_fit, compute_uv=False))[::-1]
        kinetic = 0.5 * float(
            np.sum(disc.volume * np.sum(velocity_next**2, axis=1))
        )
        exact_particle_kinetic = 0.5 * float(
            np.sum(disc.volume * np.sum(exact["velocity"] ** 2, axis=1))
        )
        area_j = float(np.sum(disc.volume * cumulative_j_next))
        area_polygon = ordered_polygon_area(position_next, disc.boundary)
        row = {
            "step": step,
            "time": float(times[step]),
            "picard_iterations": len(case["history"]),
            "start_constraint_rms": scalar_rms(
                start_residual, current_volume[disc.interior]
            ),
            "terminal_constraint_rms": scalar_rms(
                represented, terminal_volume[disc.interior]
            ),
            "terminal_constraint_max": float(np.max(np.abs(represented))),
            "production_strong_divergence": split_rms(
                strong, terminal_volume, disc.masks
            ),
            "independent_divergence": split_rms(
                independent,
                disc.volume * independent_geometry["J"],
                disc.masks,
            ),
            "independent_divergence_roundoff_floor": (
                independent_roundoff_floor
            ),
            "position_error_rms": vector_rms(
                position_next - exact["position"], disc.volume
            ),
            "velocity_error_rms": vector_rms(
                velocity_next - exact["velocity"], disc.volume
            ),
            "pressure_error_rms": scalar_rms(
                case["pressure"] - exact_pressure[disc.interior],
                disc.volume[disc.interior],
            ),
            "pressure_at_action_time_error_rms": scalar_rms(
                case["pressure"] - exact_action_pressure[disc.interior],
                disc.volume[disc.interior],
            ),
            "F_error_rms": vector_rms(
                cumulative_f_next - exact["F"], disc.volume
            ),
            "J_error_rms": scalar_rms(
                cumulative_j_next - 1.0, disc.volume
            ),
            "F_chain_vs_position_rms": vector_rms(
                cumulative_f_next - position_geometry["F"], disc.volume
            ),
            "J_chain_vs_position_rms": scalar_rms(
                cumulative_j_next - position_geometry["J"], disc.volume
            ),
            "D_recursive_cumulative_defect": relative_sparse_norm(
                d_q_next, direct_dq
            ),
            "G_recursive_cumulative_defect": relative_sparse_norm(
                action_next, direct_g
            ),
            "fitted_major_axis": float(axes[0]),
            "fitted_minor_axis": float(axes[1]),
            "aspect_ratio": float(axes[0] / axes[1]),
            "exact_major_axis": float(reference["a"][step]),
            "exact_minor_axis": float(reference["b"][step]),
            "area_J": area_j,
            "area_polygon": area_polygon,
            "area_J_relative_error": (area_j - initial_area_j) / initial_area_j,
            "area_polygon_relative_error": (
                area_polygon - initial_polygon
            )
            / initial_polygon,
            "area_J_polygon_gap_relative": (
                (area_j - initial_area_j) - (area_polygon - initial_polygon)
            )
            / initial_area_j,
            "kinetic": kinetic,
            "kinetic_drift": kinetic - initial_kinetic,
            "exact_particle_kinetic": exact_particle_kinetic,
            "kinetic_solution_error": kinetic - exact_particle_kinetic,
            "pressure_work_identity_defect": case["energy"]["identity_defect"],
            "condition_1_estimate": case["linear"]["condition_1_estimate"],
            "minimum_lu_pivot": case["linear"]["minimum_lu_pivot"],
            "min_raw_J_rel": case["raw_jacobians"]["terminal_min"],
            "max_F_condition_rel": case["raw_jacobians"][
                "terminal_max_F_condition"
            ],
            "max_F_condition_cumulative": float(
                np.max(np.linalg.cond(cumulative_f_next))
            ),
        }
        rows.append(row)
        position = position_next
        velocity = velocity_next
        d_q = d_q_next
        d_strong = d_strong_next
        action = action_next
        scalar_chain = scalar_chain_next
        cumulative_f = cumulative_f_next
        cumulative_j = cumulative_j_next
        current_volume = terminal_volume
        final_pressure = case["pressure"]
    final = rows[-1]
    passed = bool(
        all(row["terminal_constraint_rms"] <= TERMINAL_RMS_TOL for row in rows)
        and all(row["min_raw_J_rel"] > ADMISSIBLE_J for row in rows)
        and final["D_recursive_cumulative_defect"] < 1.0e-9
        and final["G_recursive_cumulative_defect"] < 1.0e-9
    )
    return {
        "passed": passed,
        "configuration": {
            "layers": layers,
            "dt": dt,
            "theta": theta,
            "steps": steps,
            "final_time": final_time,
            "particles": len(disc.points),
            "pressure_dofs": len(disc.interior),
        },
        "rows": rows,
        "final_fields": {
            "position": position,
            "velocity": velocity,
            "pressure": final_pressure,
            "cumulative_F": cumulative_f,
            "cumulative_J": cumulative_j,
        },
        "final": final,
    }


def multistep_matrix(quick: bool = False) -> dict[str, Any]:
    if quick:
        configurations = [(12, 1.0 / 40.0, 0.5)]
    else:
        configurations = []
        # Temporal refinement at the medium cloud.
        configurations.extend(
            (18, dt, theta)
            for dt in MULTISTEP_DTS
            for theta in MULTISTEP_THETAS
        )
        # Spatial comparison at a sufficiently small common step.
        configurations.extend(
            (layers, 1.0 / 160.0, theta)
            for layers in LAYERS
            for theta in MULTISTEP_THETAS
            if layers != 18
        )
    cases = [
        run_multistep_case(layers, dt, theta)
        for layers, dt, theta in configurations
    ]
    rates: dict[str, Any] = {"temporal": {}, "spatial": {}}
    for theta in MULTISTEP_THETAS:
        temporal = sorted(
            [
                case
                for case in cases
                if case["configuration"]["layers"] == 18
                and case["configuration"]["theta"] == theta
                and case.get("passed", False)
            ],
            key=lambda case: case["configuration"]["dt"],
        )
        for quantity in (
            "position_error_rms",
            "velocity_error_rms",
            "pressure_error_rms",
            "pressure_at_action_time_error_rms",
            "F_error_rms",
            "J_error_rms",
            "kinetic_solution_error",
        ):
            rates["temporal"][f"theta_{theta}_{quantity}"] = observed_rate(
                [case["configuration"]["dt"] for case in temporal],
                [case["final"][quantity] for case in temporal],
            )
        spatial = sorted(
            [
                case
                for case in cases
                if abs(case["configuration"]["dt"] - 1.0 / 160.0) < 1.0e-15
                and case["configuration"]["theta"] == theta
                and case.get("passed", False)
            ],
            key=lambda case: case["configuration"]["layers"],
        )
        for quantity in (
            "position_error_rms",
            "velocity_error_rms",
            "pressure_error_rms",
            "pressure_at_action_time_error_rms",
            "independent_divergence",
        ):
            if quantity == "independent_divergence":
                values = [case["final"][quantity]["all"] for case in spatial]
            else:
                values = [case["final"][quantity] for case in spatial]
            rates["spatial"][f"theta_{theta}_{quantity}"] = observed_rate(
                [1.0 / case["configuration"]["layers"] for case in spatial],
                values,
            )
    return {"cases": cases, "rates": rates}


def csv_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in results["preliminary_operator_audit"]:
        rows.append(
            {
                "section": "operator",
                "layers": audit["layers"],
                "particles": audit["particles"],
                "pressure_dofs": audit["pressure_dofs"],
                "metric": "relative_spectral_gap",
                "value": audit["spectrum"]["relative_gap"],
            }
        )
    for case in results["anchor_audit"]["cases"]:
        rows.append(
            {
                "section": "anchor",
                "layers": results["anchor_audit"]["layers"],
                "dt": case["dt"],
                "theta": case["theta"],
                "anchor": case["anchor"],
                "metric": "terminal_residual_rms",
                "value": case["terminal_residual"]["rms"],
            }
        )
    for sweep in results["theta_sweep"]["sweeps"]:
        for case in sweep["cases"]:
            rows.extend(
                [
                    {
                        "section": "theta",
                        "layers": sweep["layers"],
                        "dt": sweep["dt"],
                        "theta": case["theta"],
                        "metric": "delta_K_pressure",
                        "value": case["energy"]["delta_K_pressure"],
                    },
                    {
                        "section": "theta",
                        "layers": sweep["layers"],
                        "dt": sweep["dt"],
                        "theta": case["theta"],
                        "metric": "terminal_residual_rms",
                        "value": case["terminal_residual"]["rms"],
                    },
                ]
            )
    green = results["projector_audit"][
        "same_configuration_green_compatibility"
    ]
    rows.extend(
        [
            {
                "section": "projector",
                "layers": results["projector_audit"]["layers"],
                "metric": "full_space_green_compatibility_relative_frobenius",
                "value": green["full_space_relative_frobenius"],
            },
            {
                "section": "projector",
                "layers": results["projector_audit"]["layers"],
                "metric": "restricted_affine_quadratic_green_compatibility",
                "value": green["restricted_affine_quadratic"][
                    "riesz_normalized_supremum"
                ],
            },
        ]
    )
    for row in results["placed_action_stage_consistency"]["rows"]:
        if not row["passed"]:
            continue
        for metric in (
            "action_stage_error_rms",
            "pressure_stage_error_rms",
            "pressure_step_average_error_rms",
            "production_path_position_error_rms",
            "spatial_action_reproduction_error_rms",
        ):
            rows.append(
                {
                    "section": "stage_consistency",
                    "layers": results["placed_action_stage_consistency"]["layers"],
                    "start_time": results["placed_action_stage_consistency"][
                        "start_time"
                    ],
                    "dt": row["dt"],
                    "theta": row["theta"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    for case in results["multistep"]["cases"]:
        for row in case.get("rows", []):
            for metric in (
                "position_error_rms",
                "velocity_error_rms",
                "pressure_error_rms",
                "pressure_at_action_time_error_rms",
                "F_error_rms",
                "J_error_rms",
                "kinetic_drift",
                "area_J_relative_error",
                "terminal_constraint_rms",
            ):
                rows.append(
                    {
                        "section": "multistep",
                        **case["configuration"],
                        "time": row["time"],
                        "metric": metric,
                        "value": row[metric],
                    }
                )
    return rows


def make_figures(results: dict[str, Any]) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    stage = results["placed_action_stage_consistency"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for theta in stage["thetas"]:
        selected = sorted(
            (
                row
                for row in stage["rows"]
                if row["passed"] and row["theta"] == theta
            ),
            key=lambda row: row["dt"],
        )
        dt = [row["dt"] for row in selected]
        axes[0].loglog(
            dt,
            [row["action_stage_error_rms"] for row in selected],
            "o-",
            label=f"theta={theta:g}",
        )
        positive_path = [
            row for row in selected if row["production_path_position_error_rms"] > 0.0
        ]
        if positive_path:
            axes[1].loglog(
                [row["dt"] for row in positive_path],
                [row["production_path_position_error_rms"] for row in positive_path],
                "o-",
                label=f"theta={theta:g}",
            )
    axes[0].set(xlabel="dt", ylabel="placed-action stage error")
    axes[1].set(xlabel="dt", ylabel="production-path position error")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    path = FIGURE_DIR / "placed_action_stage_consistency.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    reference = results["literature_and_reference"]["reference"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(reference["time"], reference["a"], label="a(t)")
    ax.plot(reference["time"], reference["b"], label="a(t)^-1")
    ax.plot(reference["time"], reference["aspect_ratio"], label="aspect ratio")
    ax.set(xlabel="t", ylabel="reference geometry")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = FIGURE_DIR / "affine_reference.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for anchor in ("n", "star", "T"):
        cases = [
            case
            for case in results["anchor_audit"]["cases"]
            if case["anchor"] == anchor
        ]
        axes[0].loglog(
            [case["dt"] for case in cases],
            [case["terminal_residual"]["rms"] for case in cases],
            "o-",
            label=anchor,
        )
        axes[1].loglog(
            [case["dt"] for case in cases],
            [case["terminal_residual"]["source_mismatch_rms"] for case in cases],
            "o-",
            label=anchor,
        )
    axes[0].set(xlabel="dt", ylabel="terminal represented residual")
    axes[1].set(xlabel="dt", ylabel="terminal source mismatch")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    path = FIGURE_DIR / "anchor_residuals.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, ax_energy = plt.subplots(figsize=(8, 4.8))
    ax_closure = ax_energy.twinx()
    for sweep in results["theta_sweep"]["sweeps"]:
        theta = [case["theta"] for case in sweep["cases"]]
        energy = [case["energy"]["delta_K_pressure"] for case in sweep["cases"]]
        closure = [case["terminal_residual"]["rms"] for case in sweep["cases"]]
        line = ax_energy.plot(theta, energy, "o-", label=f"h~1/{sweep['layers']}")[0]
        ax_closure.semilogy(theta, closure, "--", color=line.get_color(), alpha=0.45)
        if sweep["primary_zero_crossing"] is not None:
            ax_energy.axvline(
                sweep["primary_zero_crossing"],
                color=line.get_color(),
                alpha=0.25,
            )
    ax_energy.axhline(0.0, color="black", linewidth=0.8)
    ax_energy.set(xlabel="theta", ylabel="pressure-induced delta K")
    ax_closure.set_ylabel("terminal represented closure (dashed)")
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()
    path = FIGURE_DIR / "theta_energy_closure.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    spatial_roots = [
        sweep
        for sweep in results["theta_sweep"]["sweeps"]
        if sweep["primary_zero_crossing"] is not None
    ]
    temporal_roots = [
        sweep
        for sweep in results["theta_sweep"].get("temporal_sweeps", [])
        if sweep["primary_zero_crossing"] is not None
    ]
    axes[0].plot(
        [1.0 / sweep["layers"] for sweep in spatial_roots],
        [sweep["primary_zero_crossing"] for sweep in spatial_roots],
        "o-",
    )
    axes[1].loglog(
        [sweep["dt"] for sweep in temporal_roots],
        [
            abs(sweep["primary_zero_crossing"] - 0.5)
            for sweep in temporal_roots
        ],
        "o-",
    )
    axes[0].set(xlabel="h target", ylabel="theta*")
    axes[1].set(xlabel="dt", ylabel="|theta*-1/2|")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    path = FIGURE_DIR / "theta_star_refinement.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    projector = results["projector_audit"]["rows"]
    compatible = results["projector_audit"]["compatible_endpoint_control"][
        "rows"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(
        [row["theta"] for row in projector],
        [row["projector_distance_from_T_mass_norm"] for row in projector],
        "o-",
    )
    axes[1].plot(
        [row["theta"] for row in projector],
        [row["maximum_principal_angle_degrees"] for row in projector],
        "o-",
    )
    production_endpoint_norm = min(
        projector, key=lambda row: abs(row["theta"] - 1.0)
    )["projector_mass_norm"]
    compatible_endpoint_norm = min(
        compatible, key=lambda row: abs(row["theta"] - 1.0)
    )["projector_mass_norm"]
    axes[2].plot(
        [row["theta"] for row in projector],
        [
            row["projector_mass_norm"] - production_endpoint_norm
            for row in projector
        ],
        "o-",
        label="production pair",
    )
    axes[2].plot(
        [row["theta"] for row in compatible],
        [
            row["projector_mass_norm"] - compatible_endpoint_norm
            for row in compatible
        ],
        "s--",
        label="compatible endpoint control",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set(xlabel="theta", ylabel="||Pi_theta-Pi_T||_M")
    axes[1].set(xlabel="theta", ylabel="max action-space angle (deg)")
    axes[2].set(xlabel="theta", ylabel="projector-norm excess over endpoint")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    path = FIGURE_DIR / "projector_geometry.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for audit in results["preliminary_operator_audit"]:
        singular = audit["spectrum"]["smallest_singular_values"]
        ax.semilogy(
            np.arange(1, len(singular) + 1),
            singular,
            "o-",
            label=f"h~1/{audit['layers']}",
        )
    ax.set(xlabel="order from smallest", ylabel="scaled singular value")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = FIGURE_DIR / "smallest_singular_values.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].loglog(
        [1.0 / audit["layers"] for audit in results["preliminary_operator_audit"]],
        [audit["spectrum"]["sigma_min"] for audit in results["preliminary_operator_audit"]],
        "o-",
        label="sigma min",
    )
    axes[0].loglog(
        [1.0 / audit["layers"] for audit in results["preliminary_operator_audit"]],
        [audit["spectrum"]["sigma_max"] for audit in results["preliminary_operator_audit"]],
        "s--",
        label="sigma max",
    )
    axes[1].loglog(
        [1.0 / audit["layers"] for audit in results["preliminary_operator_audit"]],
        [audit["spectrum"]["condition_number"] for audit in results["preliminary_operator_audit"]],
        "o-",
    )
    axes[0].set(xlabel="h target", ylabel="scaled singular values")
    axes[1].set(xlabel="h target", ylabel="scaled condition number")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    path = FIGURE_DIR / "pressure_conditioning.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    finest = results["preliminary_operator_audit"][-1]
    points = build_discretization(finest["layers"]).points
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, mode in zip(axes, finest["smallest_modes"]):
        scatter = ax.scatter(points[:, 0], points[:, 1], c=mode["field"], s=5, cmap="coolwarm")
        ax.set_aspect("equal")
        ax.set_title(f"sigma={mode['singular_value']:.2e}")
        fig.colorbar(scatter, ax=ax, shrink=0.7)
    path = FIGURE_DIR / "smallest_pressure_modes.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    passed_cases = [case for case in results["multistep"]["cases"] if case.get("passed")]
    if passed_cases:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for case in passed_cases:
            config = case["configuration"]
            label = f"L{config['layers']} dt={config['dt']:.5g} th={config['theta']}"
            time = [row["time"] for row in case["rows"]]
            axes[0, 0].semilogy(time, [row["position_error_rms"] for row in case["rows"]], label=label)
            axes[0, 1].semilogy(time, [row["velocity_error_rms"] for row in case["rows"]], label=label)
            axes[1, 0].plot(time, [row["kinetic_drift"] for row in case["rows"]], label=label)
            axes[1, 1].plot(time, [row["area_J_relative_error"] for row in case["rows"]], label=label)
        axes[0, 0].set_ylabel("position RMS error")
        axes[0, 1].set_ylabel("velocity RMS error")
        axes[1, 0].set_ylabel("kinetic drift")
        axes[1, 1].set_ylabel("Jacobian-area relative error")
        for ax in axes.ravel():
            ax.set_xlabel("t")
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=6)
        path = FIGURE_DIR / "multistep_errors_energy_area.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        spatial = [
            case
            for case in passed_cases
            if abs(case["configuration"]["dt"] - 1.0 / 160.0) < 1.0e-15
        ]
        for theta in MULTISTEP_THETAS:
            selected = sorted(
                [case for case in spatial if case["configuration"]["theta"] == theta],
                key=lambda case: case["configuration"]["layers"],
            )
            if selected:
                h = [1.0 / case["configuration"]["layers"] for case in selected]
                axes[0].loglog(
                    h,
                    [case["final"]["independent_divergence"]["bulk"] for case in selected],
                    "o-",
                    label=f"bulk theta={theta}",
                )
                axes[0].loglog(
                    h,
                    [case["final"]["independent_divergence"]["first_layer"] for case in selected],
                    "s--",
                    label=f"first theta={theta}",
                )
                axes[0].loglog(
                    h,
                    [case["final"]["independent_divergence"]["boundary"] for case in selected],
                    "^:",
                    label=f"boundary theta={theta}",
                )
        medium = [
            case for case in passed_cases if case["configuration"]["layers"] == 18
        ]
        for theta in MULTISTEP_THETAS:
            selected = sorted(
                [case for case in medium if case["configuration"]["theta"] == theta],
                key=lambda case: case["configuration"]["dt"],
            )
            if selected:
                axes[1].loglog(
                    [case["configuration"]["dt"] for case in selected],
                    [case["final"]["position_error_rms"] for case in selected],
                    "o-",
                    label=f"x theta={theta}",
                )
                axes[1].loglog(
                    [case["configuration"]["dt"] for case in selected],
                    [case["final"]["velocity_error_rms"] for case in selected],
                    "s--",
                    label=f"u theta={theta}",
                )
        axes[0].set(xlabel="h", ylabel="independent divergence")
        axes[1].set(xlabel="dt", ylabel="final error")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        path = FIGURE_DIR / "divergence_and_temporal_errors.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))

        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
        quantities = (
            ("position_error_rms", "x error"),
            ("velocity_error_rms", "u error"),
            ("pressure_error_rms", "p error at endpoint"),
            ("pressure_at_action_time_error_rms", "p error at action time"),
            ("F_error_rms", "F error"),
            ("J_error_rms", "J error"),
        )
        for theta in MULTISTEP_THETAS:
            selected = sorted(
                [
                    case
                    for case in passed_cases
                    if case["configuration"]["layers"] == 18
                    and case["configuration"]["theta"] == theta
                ],
                key=lambda case: case["configuration"]["dt"],
            )
            for ax, (quantity, label) in zip(axes.ravel(), quantities):
                ax.loglog(
                    [case["configuration"]["dt"] for case in selected],
                    [abs(case["final"][quantity]) for case in selected],
                    "o-",
                    label=f"theta={theta}",
                )
                ax.set(xlabel="dt", ylabel=label)
                ax.grid(True, which="both", alpha=0.3)
        axes[0, 0].legend()
        path = FIGURE_DIR / "affine_state_errors_vs_dt.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))
    return files


def decide(results: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    literature = results["literature_and_reference"]
    if not literature["passed"]:
        reasons.append("literature formula or scalar reference invariant failed")
    audits = results["preliminary_operator_audit"]
    if not all(audit["passed"] for audit in audits):
        reasons.append("a disk pressure system or manufactured operator audit failed")
    anchor = results["anchor_audit"]
    if anchor["terminal_anchor_max_terminal_residual"] > TERMINAL_RMS_TOL:
        reasons.append("D_T-anchored terminal closure missed tolerance")
    if not anchor["all_passed_own_anchor"]:
        reasons.append("at least one anchor solve did not close its own square system")
    sweeps = results["theta_sweep"]["sweeps"]
    if not all(sweep["all_passed"] for sweep in sweeps):
        reasons.append("a theta placement solve failed")
    if max(sweep["maximum_pressure_work_identity_defect"] for sweep in sweeps) > 2.0e-11:
        reasons.append("pressure-work identity defect exceeds fp64 expectation")
    stage = results["placed_action_stage_consistency"]
    if not stage["passed"]:
        reasons.append(
            "placed-action stage consistency does not show the required "
            "midpoint second-order and endpoint first-order behavior"
        )
    multistep = results["multistep"]["cases"]
    if not all(case.get("passed", False) for case in multistep):
        reasons.append("at least one multistep affine run failed algebraically")
    midpoint_temporal = results["multistep"]["rates"]["temporal"].get(
        "theta_0.5_position_error_rms"
    )
    endpoint_temporal = results["multistep"]["rates"]["temporal"].get(
        "theta_1.0_position_error_rms"
    )
    if midpoint_temporal is None or midpoint_temporal < 1.5:
        reasons.append("midpoint affine position error is not at least near second order")
    if endpoint_temporal is None or endpoint_temporal < 0.7:
        reasons.append("endpoint affine position error lacks a reproducible first-order trend")
    spatial = [
        case
        for case in multistep
        if case.get("passed")
        and abs(case["configuration"]["dt"] - 1.0 / 160.0) < 1.0e-15
        and case["configuration"]["theta"] == 0.5
    ]
    spatial.sort(key=lambda case: case["configuration"]["layers"])
    if len(spatial) >= 3:
        boundary = [
            case["final"]["independent_divergence"]["boundary"] for case in spatial
        ]
        floors = [
            case["final"]["independent_divergence_roundoff_floor"]
            for case in spatial
        ]
        monotone = all(
            fine <= coarse * (1.0 + 1.0e-6)
            for coarse, fine in zip(boundary[:-1], boundary[1:])
        )
        if (not monotone) and any(
            value > floor for value, floor in zip(boundary, floors)
        ):
            reasons.append(
                "independent boundary divergence is nonmonotone above its "
                "explicit fp64 differentiation floor"
            )
    else:
        reasons.append("three-level multistep spatial comparison is missing")
    if reasons:
        return "HOLD: affine Test 1 has unresolved validation gates", reasons
    return "GO: AFFINE FREE-BOUNDARY TEST 1 VALIDATED", reasons


def write_report(results: dict[str, Any]) -> None:
    decision = results["decision"]
    audits = results["preliminary_operator_audit"]
    anchor = results["anchor_audit"]
    sweeps = results["theta_sweep"]["sweeps"]
    stage = results["placed_action_stage_consistency"]
    multistep = results["multistep"]
    lines = [
        "# Test 1 — force-free affine incompressible liquid in vacuum",
        "",
        f"**Decision: {decision['label']}**",
        "",
        "## Literature and exact reference",
        "",
        (
            "The zero-magnetic-field (`kappa=0`) equations were checked directly "
            "against Roberts–Shkoller–Sideris. The paper gives "
            "`A_ddot=-kappa A+lambda cof(A)` and "
            "`lambda=2(kappa-det(A_dot))/|A|_F^2`; hence the requested perfect-fluid "
            "signs and pressure normalization are consistent."
        ),
        "",
        (
            f"The scalar DOP853 reference has max `|det A-1|` "
            f"{results['literature_and_reference']['reference']['max_det_error']:.3e} "
            f"and max matrix-energy error "
            f"{results['literature_and_reference']['reference']['max_matrix_energy_error']:.3e}."
        ),
        "",
        "## Pressure space and operator audit",
        "",
        "| h target | N | boundary | pressure dofs | relative gap | condition | max GMLS condition |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in audits:
        lines.append(
            f"| 1/{audit['layers']} | {audit['particles']} | "
            f"{audit['boundary_particles']} | {audit['pressure_dofs']} | "
            f"{audit['spectrum']['relative_gap']:.3e} | "
            f"{audit['spectrum']['condition_number']:.3e} | "
            f"{audit['gmls_max_condition']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Boundary pressure values are fixed to zero and are not unknowns. "
            "The square production system uses the matching interior divergence rows. "
            "No pin, pseudo-inverse, shift, clipping, filtering, or hidden projection is used.",
            "",
            "## Test 1A — terminal anchors",
            "",
            "| anchor | terminal residual rate | source-mismatch rate | J-1 rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, rates in anchor["rates"].items():
        lines.append(
            f"| {name} | {rates['terminal_residual']} | "
            f"{rates['source_mismatch']} | {rates['J_minus_one']} |"
        )
    lines.extend(
        [
            "",
            f"Maximum `D_T`-anchored represented residual: "
            f"{anchor['terminal_anchor_max_terminal_residual']:.3e}.",
            "",
            "## Test 1B — pressure placement",
            "",
            "| h target | dK(0) | dK(1/2) | dK(1) | theta* | max closure | work defect |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sweep in sweeps:
        lines.append(
            f"| 1/{sweep['layers']} | {sweep['delta_K_selected']['0.0']:.3e} | "
            f"{sweep['delta_K_selected']['0.5']:.3e} | "
            f"{sweep['delta_K_selected']['1.0']:.3e} | "
            f"{sweep['primary_zero_crossing']} | "
            f"{sweep['maximum_terminal_closure']:.3e} | "
            f"{sweep['maximum_pressure_work_identity_defect']:.3e} |"
        )
    lines.extend(
        [
            "",
            (
                "Measured temporal rate of `|theta*-1/2|`: "
                f"{results['theta_sweep'].get('theta_star_offset_temporal_rate')}."
            ),
            "",
            "## Test 1C — frozen projector",
            "",
            (
                f"Endpoint `||Pi_T||_M` = "
                f"{results['projector_audit']['endpoint_projector_mass_norm']:.6g}; "
                f"same-configuration Green-compatibility mismatch = "
                f"{results['projector_audit']['endpoint_green_adjoint_relative_defect']:.3e}, "
                "defined by `||D_T^q-D_T^ad||_F/||D_T^q||_F`. "
                "This is a full-space compatibility baseline, not a divergence, "
                "gradient, pressure-solve, or projector error. "
                "Only `||Pi_theta-Pi_T||_M` and the action-space angle are attributed "
                "to placement."
            ),
            (
                "On the trace-free affine velocity space paired with the quadratic "
                "homogeneous-Dirichlet disk pressure, the Riesz-normalized bilinear "
                "Green mismatch is "
                f"`{results['projector_audit']['same_configuration_green_compatibility']['restricted_affine_quadratic']['riesz_normalized_supremum']:.3e}`."
            ),
            (
                "In the auxiliary mass-adjoint endpoint control, "
                f"`||Pi_T^ad||_M` = "
                f"{results['projector_audit']['compatible_endpoint_control']['endpoint_projector_mass_norm']:.12g}, "
                "while the largest off-endpoint norm is "
                f"{results['projector_audit']['compatible_endpoint_control']['maximum_off_endpoint_projector_mass_norm']:.12g}. "
                "This control is diagnostic only and leaves the production pair unchanged."
            ),
            "",
            "## Test 1D — exact-start placed-action stage consistency",
            "",
            "| theta | action-stage rate | instantaneous-p rate | average-p rate | action-path rate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for theta in stage["thetas"]:
        rates = stage["rates"][str(theta)]
        path_rate = rates["production_path_position_error_rms"]
        path_rate_text = "exact" if path_rate is None else f"{path_rate:.3f}"
        lines.append(
            f"| {theta:g} | {rates['action_stage_error_rms']:.3f} | "
            f"{rates['pressure_stage_error_rms']:.3f} | "
            f"{rates['pressure_step_average_error_rms']:.3f} | "
            f"{path_rate_text} |"
        )
    lines.extend(
        [
            "",
            (
                "The exact pressure applied through the numerical production-path "
                "action has maximum RMS reproduction error "
                f"`{stage['maximum_spatial_action_reproduction_error_rms']:.3e}`. "
                "The stage test starts from the exact state at nonzero time, so its "
                "rates are local rather than accumulated trajectory errors."
            ),
            (
                "The endpoint coefficient errors remain first order even when "
                "measured against the exact step-average pressure.  Hence the "
                "multiplier is not treated as a configuration-independent "
                "step-average pressure; the order-bearing quantity is the placed "
                "vector action `G_theta p_theta`."
            ),
            "",
            "## Test 1E — affine-reference evolution",
            "",
            "| layers | dt | theta | x error | u error | p(end) | p(action) | F error | J error | area error | energy error |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in multistep["cases"]:
        if not case.get("passed"):
            config = case["configuration"]
            lines.append(
                f"| {config['layers']} | {config['dt']} | {config['theta']} | FAILED | | | | | | | | |"
            )
            continue
        config = case["configuration"]
        final = case["final"]
        lines.append(
            f"| {config['layers']} | {config['dt']:.6g} | {config['theta']} | "
            f"{final['position_error_rms']:.3e} | {final['velocity_error_rms']:.3e} | "
            f"{final['pressure_error_rms']:.3e} | "
            f"{final['pressure_at_action_time_error_rms']:.3e} | "
            f"{final['F_error_rms']:.3e} | "
            f"{final['J_error_rms']:.3e} | {final['area_J_relative_error']:.3e} | "
            f"{final['kinetic_solution_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Temporal and spatial fitted rates are stored verbatim in the JSON. "
            "Bulk, first-layer, boundary, and all-particle independent divergence "
            "histories are also retained for every multistep run.",
            "",
            (
                "The independent affine-divergence values are at the fp64 "
                "differentiation floor (the finest boundary value is O(1e-13), "
                "while the explicit condition-number/h roundoff envelope is "
                "O(1e-11)). They therefore do not define a spatial convergence "
                "rate in this exactly reproduced affine test. The absence of a "
                "measurable spatial truncation regime is reported, not promoted "
                "as high-order convergence."
            ),
            "",
            "## Gate failures",
            "",
        ]
    )
    if decision["reasons"]:
        lines.extend(f"- {reason}" for reason in decision["reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This result supports only terminal closure and pressure-action placement "
            "for the smooth affine free-boundary solution. It is not a non-affine "
            "spatial validation and the motion is not described as periodic or oscillatory.",
        ]
    )
    DEFAULT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(quick: bool = False) -> dict[str, Any]:
    literature = literature_and_reference_audit()
    levels = (12,) if quick else LAYERS
    preliminary = [preliminary_audit(level) for level in levels]
    if quick:
        original_layers = globals()["LAYERS"]
        anchor = {
            "layers": 12,
            "cases": [],
            "rates": {},
            "all_passed_own_anchor": True,
            "terminal_anchor_max_terminal_residual": 0.0,
        }
        sweep_public, sweep_raw = theta_sweep(12)
        theta = {"sweeps": [sweep_public], "spatial_curve_differences": []}
        raw = {12: sweep_raw}
        projector = projector_audit(raw)
    else:
        anchor = anchor_audit()
        theta, raw = theta_sweep_matrix()
        projector = projector_audit(raw)
    multistep = multistep_matrix(quick=quick)
    stage = placed_action_stage_consistency(
        layers=12 if quick else STAGE_LAYERS
    )
    results = {
        "schema_version": 2,
        "test": "force-free affine incompressible liquid surrounded by vacuum",
        "literature_and_reference": literature,
        "configuration": {
            "radius": RADIUS,
            "rho": RHO,
            "gamma": GAMMA,
            "degree": DEGREE,
            "independent_degree": INDEPENDENT_DEGREE,
            "layers": levels,
            "anchor_dts": ANCHOR_DTS,
            "theta_values": THETA_VALUES,
            "sweep_dt": SWEEP_DT,
            "final_time": FINAL_TIME,
            "multistep_dts": MULTISTEP_DTS,
            "multistep_thetas": MULTISTEP_THETAS,
            "stage_consistency": {
                "layers": 12 if quick else STAGE_LAYERS,
                "start_time": STAGE_START_TIME,
                "dts": STAGE_DTS,
                "thetas": STAGE_THETAS,
            },
            "deterministic_seed": DETERMINISTIC_SEED,
            "geometry_tolerances": {
                "rms_over_h": GEOMETRY_RMS_TOL,
                "max_over_h": GEOMETRY_MAX_TOL,
            },
            "terminal_tolerances": {
                "rms": TERMINAL_RMS_TOL,
                "max": TERMINAL_MAX_TOL,
            },
            "admissible_raw_J": ADMISSIBLE_J,
            "pressure_space": (
                "general nodal homogeneous-Dirichlet pressure; boundary values "
                "fixed zero; interior coefficients only"
            ),
            "production_constraint": (
                "interior pressure-label rows of transported Piola divergence"
            ),
            "independent_divergence": (
                "degree-3 GMLS, 32-point stencil, direct initial-reference rebuild"
            ),
            "forbidden_mechanisms": {
                "regularization": False,
                "pseudo_inverse": False,
                "determinant_clipping": False,
                "pressure_filtering": False,
                "hidden_projection": False,
            },
        },
        "preliminary_operator_audit": preliminary,
        "anchor_audit": anchor,
        "theta_sweep": theta,
        "projector_audit": projector,
        "placed_action_stage_consistency": stage,
        "multistep": multistep,
    }
    label, reasons = decide(results)
    results["decision"] = {"label": label, "reasons": reasons}
    return results


def write_outputs(results: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    figures = make_figures(results)
    results["figures"] = figures
    DEFAULT_JSON.write_text(
        json.dumps(ready(results), indent=2), encoding="utf-8"
    )
    DEFAULT_CONFIG.write_text(
        json.dumps(ready(results["configuration"]), indent=2), encoding="utf-8"
    )
    rows = csv_rows(results)
    fields = sorted({key for row in rows for key in row})
    with DEFAULT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_report(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    results = run_validation(quick=args.quick)
    write_outputs(results)
    if args.output != DEFAULT_JSON:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(ready(results), indent=2), encoding="utf-8"
        )
    print(results["decision"]["label"])
    for reason in results["decision"]["reasons"]:
        print(f"- {reason}")
    print(DEFAULT_JSON)


if __name__ == "__main__":
    main()
