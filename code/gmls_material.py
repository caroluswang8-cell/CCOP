"""Sparse, arbitrary-order material GMLS derivatives for point-cloud CCOP."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree


def total_degree_exponents(degree: int, dimension: int = 2) -> list[tuple[int, ...]]:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if dimension != 2:
        raise NotImplementedError("the current CCOP realization is two-dimensional")
    return [(a, total - a) for total in range(degree + 1) for a in range(total, -1, -1)]


def _monomial_matrix(local: np.ndarray, exponents: Iterable[tuple[int, int]]) -> np.ndarray:
    return np.column_stack(
        [local[:, 0] ** a * local[:, 1] ** b for a, b in exponents]
    )


def _wendland_c2(radius: np.ndarray) -> np.ndarray:
    clipped = np.clip(1.0 - radius, 0.0, None)
    return clipped**4 * (4.0 * radius + 1.0)


@dataclass(frozen=True)
class GMLSDiagnostics:
    degree: int
    basis_size: int
    stencil_size: int
    maximum_condition: float
    maximum_moment_defect: float
    minimum_scaled_singular_value: float


class MaterialGMLS:
    """One material derivative object shared by geometry, divergence, and pressure.

    Neighbor relations and differentiation weights live on the material reference
    cloud. Configuration dependence enters only through fields reconstructed with
    these fixed sparse derivative matrices.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        degree: int = 2,
        stencil_size: int | None = None,
        stencil_factor: float = 2.5,
        periodic_box: tuple[float, float] | None = None,
        relative_svd_cutoff: float = 1.0e-12,
        neighbors: np.ndarray | None = None,
    ) -> None:
        points = np.asarray(reference_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("reference_points must have shape (N, 2)")
        if len(points) < 3:
            raise ValueError("at least three material points are required")
        self.X = points
        self.n = len(points)
        self.degree = int(degree)
        self.exponents = total_degree_exponents(self.degree)
        self.basis_size = len(self.exponents)
        requested = stencil_size
        if neighbors is not None:
            supplied_neighbors = np.asarray(neighbors, dtype=np.int32)
            if supplied_neighbors.ndim != 2 or supplied_neighbors.shape[0] != self.n:
                raise ValueError("neighbors must have shape (N, stencil_size)")
            if np.any(supplied_neighbors < 0) or np.any(supplied_neighbors >= self.n):
                raise ValueError("neighbor index outside the material cloud")
            self.stencil_size = supplied_neighbors.shape[1]
            if self.stencil_size < self.basis_size + 2:
                raise ValueError("supplied neighbor stencil is too small")
        else:
            if requested is None:
                requested = int(np.ceil(stencil_factor * self.basis_size))
            self.stencil_size = min(self.n, max(self.basis_size + 2, int(requested)))
        self.periodic_box = None if periodic_box is None else np.asarray(periodic_box, dtype=np.float64)
        self.relative_svd_cutoff = float(relative_svd_cutoff)
        if neighbors is not None:
            self.neighbors = supplied_neighbors.copy()
        elif self.periodic_box is not None:
            if self.periodic_box.shape != (2,) or np.any(self.periodic_box <= 0.0):
                raise ValueError("periodic_box must contain two positive lengths")
            wrapped = np.mod(points, self.periodic_box)
            if not np.allclose(wrapped, points, atol=1.0e-13, rtol=0.0):
                raise ValueError("periodic reference points must lie in [0, L) in each direction")
            tree = cKDTree(points, boxsize=self.periodic_box)
        else:
            tree = cKDTree(points)
        if neighbors is None:
            _, queried_neighbors = tree.query(points, k=self.stencil_size)
            if self.stencil_size == 1:
                queried_neighbors = queried_neighbors[:, None]
            self.neighbors = np.asarray(queried_neighbors, dtype=np.int32)

        self.Hx, diag_x, self.wx = self._assemble_derivative((1, 0))
        self.Hy, diag_y, self.wy = self._assemble_derivative((0, 1))
        self.diagnostics = GMLSDiagnostics(
            degree=self.degree,
            basis_size=self.basis_size,
            stencil_size=self.stencil_size,
            maximum_condition=max(diag_x[0], diag_y[0]),
            maximum_moment_defect=max(diag_x[1], diag_y[1]),
            minimum_scaled_singular_value=min(diag_x[2], diag_y[2]),
        )

    def _relative_offsets(self, center: int, neighbors: np.ndarray) -> np.ndarray:
        delta = self.X[neighbors] - self.X[center]
        if self.periodic_box is not None:
            delta -= self.periodic_box * np.round(delta / self.periodic_box)
        return delta

    def _functional(self, derivative: tuple[int, int], scale: float) -> np.ndarray:
        functional = np.zeros(self.basis_size, dtype=np.float64)
        try:
            column = self.exponents.index(tuple(derivative))
        except ValueError as exc:
            raise ValueError("derivative order exceeds the polynomial degree") from exc
        functional[column] = (
            factorial(derivative[0])
            * factorial(derivative[1])
            / scale ** sum(derivative)
        )
        return functional

    def _assemble_derivative(
        self, derivative: tuple[int, int]
    ) -> tuple[csr_matrix, tuple[float, float, float], np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        maximum_condition = 0.0
        maximum_defect = 0.0
        minimum_scaled_singular = np.inf
        padded_weights = np.zeros((self.n, self.stencil_size), dtype=np.float64)

        for i, neighbors in enumerate(self.neighbors):
            delta = self._relative_offsets(i, neighbors)
            distance = np.linalg.norm(delta, axis=1)
            scale = max(float(np.max(distance)), np.finfo(float).eps)
            support = 1.05 * scale
            local = delta / scale
            basis = _monomial_matrix(local, self.exponents)
            kernel = _wendland_c2(distance / support)
            weighted_basis = np.sqrt(kernel)[:, None] * basis
            singular = np.linalg.svd(weighted_basis, compute_uv=False)
            cutoff = self.relative_svd_cutoff * singular[0]
            retained = singular[singular > cutoff]
            if len(retained) < self.basis_size:
                raise np.linalg.LinAlgError(
                    f"rank-deficient GMLS stencil at particle {i}: "
                    f"rank={len(retained)}, basis={self.basis_size}"
                )
            scaled_min = float(retained[-1] / retained[0])
            condition = float(retained[0] / retained[-1])
            functional = self._functional(derivative, scale)
            gram = basis.T @ (kernel[:, None] * basis)
            coefficient = np.linalg.solve(gram, functional)
            weights = kernel * (basis @ coefficient)
            padded_weights[i] = weights
            defect = float(np.linalg.norm(basis.T @ weights - functional))

            rows.extend([i] * len(neighbors))
            cols.extend(neighbors.tolist())
            data.extend(weights.tolist())
            maximum_condition = max(maximum_condition, condition)
            maximum_defect = max(maximum_defect, defect)
            minimum_scaled_singular = min(minimum_scaled_singular, scaled_min)

        matrix = coo_matrix((data, (rows, cols)), shape=(self.n, self.n)).tocsr()
        return matrix, (maximum_condition, maximum_defect, minimum_scaled_singular), padded_weights

    def gradient(self, scalar: np.ndarray) -> np.ndarray:
        value = np.asarray(scalar, dtype=np.float64).reshape(self.n)
        return np.column_stack((self.Hx @ value, self.Hy @ value))

    def material_divergence(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64).reshape(self.n, 2)
        return np.asarray(self.Hx @ value[:, 0] + self.Hy @ value[:, 1]).reshape(-1)
