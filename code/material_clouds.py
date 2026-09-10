"""Material particle clouds and geometry diagnostics used by the SISC path."""

from __future__ import annotations

import numpy as np


def ordered_polygon_area(points: np.ndarray, boundary: np.ndarray) -> float:
    polygon = np.asarray(points, dtype=np.float64)[
        np.asarray(boundary, dtype=np.int32)
    ]
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * abs(float(np.sum(
        polygon[:, 0] * following[:, 1]
        - following[:, 0] * polygon[:, 1]
    )))


def square_material_cloud(
    n_side: int,
    half_side: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor-product square cloud with an ordered material boundary."""
    if n_side < 5:
        raise ValueError("n_side must be at least five")
    coordinates = np.linspace(-half_side, half_side, n_side)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))

    def index(i: int, j: int) -> int:
        return i * n_side + j

    boundary = np.asarray(
        [index(i, 0) for i in range(n_side)]
        + [index(n_side - 1, j) for j in range(1, n_side)]
        + [index(i, n_side - 1) for i in range(n_side - 2, -1, -1)]
        + [index(0, j) for j in range(n_side - 2, 0, -1)],
        dtype=np.int32,
    )
    dx = 2.0 * half_side / (n_side - 1)
    tensor_weight = np.ones((n_side, n_side), dtype=np.float64)
    tensor_weight[[0, -1], :] *= 0.5
    tensor_weight[:, [0, -1]] *= 0.5
    return points, boundary, dx * dx * tensor_weight.ravel()


def ring_disk_material_cloud(
    radius: float,
    spacing: float,
    boundary_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concentric material disk with a strict ordered free-surface ring."""
    if radius <= 0.0 or spacing <= 0.0:
        raise ValueError("radius and spacing must be positive")
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    ring_count = int(np.floor(radius / spacing))
    for level in range(1, ring_count + 1):
        radial = level * spacing
        if radial >= radius:
            break
        count = max(int(np.round(2.0 * np.pi * radial / spacing)), 6)
        if count % 2:
            count += 1
        phase = 0.5 * (level % 2) * 2.0 * np.pi / count
        angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + phase
        points.extend(zip(radial * np.cos(angle), radial * np.sin(angle)))

    if boundary_points is None:
        boundary_points = max(
            int(np.round(2.0 * np.pi * radius / spacing)), 32
        )
        if boundary_points % 2:
            boundary_points += 1
    angle = np.linspace(
        0.0, 2.0 * np.pi, int(boundary_points), endpoint=False
    )
    start = len(points)
    points.extend(zip(radius * np.cos(angle), radius * np.sin(angle)))
    cloud = np.asarray(points, dtype=np.float64)
    boundary = np.arange(start, len(cloud), dtype=np.int32)

    radii = np.linalg.norm(cloud, axis=1)
    tolerance = 1.0e-8 * max(radius, 1.0)
    levels = np.unique(np.round(radii / tolerance) * tolerance)
    interfaces = np.empty(len(levels) + 1, dtype=np.float64)
    interfaces[0] = 0.0
    interfaces[-1] = radius
    interfaces[1:-1] = 0.5 * (levels[:-1] + levels[1:])
    volume = np.empty(len(cloud), dtype=np.float64)
    for level, inner, outer in zip(
        levels, interfaces[:-1], interfaces[1:]
    ):
        members = np.flatnonzero(
            np.isclose(radii, level, rtol=0.0, atol=0.51 * tolerance)
        )
        volume[members] = (
            np.pi * (outer * outer - inner * inner) / len(members)
        )
    volume *= np.pi * radius * radius / np.sum(volume)
    return cloud, boundary, volume
