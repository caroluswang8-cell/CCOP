"""Automated regression gates for the affine free-boundary Test 1."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import test1_affine_free_boundary as affine


def _representative_step() -> dict:
    disc = affine.build_discretization(12)
    velocity = np.column_stack((disc.points[:, 0], -disc.points[:, 1]))
    identity = np.broadcast_to(
        np.eye(2), (len(disc.points), 2, 2)
    ).copy()
    return affine.solve_step(
        disc,
        disc.points,
        velocity,
        0.02,
        0.5,
        disc.D_q0,
        disc.D_strong0,
        disc.G0,
        identity,
        disc.volume,
        anchor="T",
    )


def test_reference_ode_invariants() -> None:
    reference = affine.affine_reference(np.linspace(0.0, 0.5, 101))
    assert np.max(np.abs(reference["det_A"] - 1.0)) < 1.0e-14
    assert np.max(np.abs(reference["matrix_energy"] - 1.0)) < 5.0e-13
    assert np.max(reference["matrix_equation_defect"]) < 1.0e-12


def test_exact_boundary_pressure() -> None:
    disc = affine.build_discretization(12)
    pressure = 0.5 * (1.0 - np.sum(disc.points**2, axis=1))
    assert np.max(np.abs(pressure[disc.boundary])) < 1.0e-13
    assert pressure[np.argmin(np.sum(disc.points**2, axis=1))] > 0.0


def test_affine_map_reconstruction() -> None:
    disc = affine.build_discretization(12)
    matrix = np.diag((1.2, 1.0 / 1.2))
    mapped = disc.points @ matrix.T
    geometry = affine.relative_geometry(
        disc.derivative,
        mapped - disc.points,
        np.broadcast_to(np.eye(2), (len(disc.points), 2, 2)),
    )
    assert affine.vector_rms(
        geometry["F"] - matrix, disc.volume
    ) < 1.0e-12
    assert np.max(np.abs(geometry["J"] - 1.0)) < 1.0e-11


def test_pressure_system_rank() -> None:
    disc = affine.build_discretization(12)
    spectrum = affine.sparse_spectrum(
        disc.D_q0,
        disc.G0,
        disc.volume[disc.interior],
        disc.volume[disc.interior],
    )
    assert spectrum["rank_at_1e-12"] == len(disc.interior)
    assert spectrum["sigma_min"] > 0.0


def test_terminal_closure_and_work_identity() -> None:
    case = _representative_step()
    assert case["passed"]
    assert case["terminal_residual"]["rms"] < 1.0e-12
    assert case["terminal_residual"]["max"] < 1.0e-11
    assert case["energy"]["identity_defect"] < 1.0e-13
    assert case["raw_jacobians"]["terminal_min"] > affine.ADMISSIBLE_J


def test_projector_algebra() -> None:
    case = _representative_step()
    d_q = case["D_q_terminal"]
    action = case["G_theta"]
    response = (d_q @ action).tocsc()
    lu = spla.splu(response)
    projector = np.eye(d_q.shape[1]) - action @ lu.solve(d_q.toarray())
    assert (
        np.linalg.norm(d_q @ projector) / spla.norm(d_q)
        < 2.0e-13
    )
    assert (
        np.linalg.norm(projector @ projector - projector)
        / np.linalg.norm(projector)
        < 2.0e-13
    )
    assert (
        np.linalg.norm(projector @ action.toarray())
        / spla.norm(action)
        < 2.0e-13
    )


def test_compatible_endpoint_projector_is_mass_orthogonal() -> None:
    disc = affine.build_discretization(12)
    case = _representative_step()
    terminal = case["terminal_geometry"]
    action = affine.transport_action_sparse(
        disc.G0, terminal["inverse"], terminal["J"]
    )
    test_mass = disc.volume[disc.interior] * terminal["J"][disc.interior]
    velocity_mass = np.repeat(disc.volume, 2)
    divergence = (
        -sp.diags(1.0 / test_mass)
        @ action.T
        @ sp.diags(velocity_mass)
    ).tocsr()
    response = (divergence @ action).tocsc()
    lu = spla.splu(response)
    projector = np.eye(divergence.shape[1]) - action @ lu.solve(
        divergence.toarray()
    )
    root_mass = np.sqrt(velocity_mass)
    weighted = root_mass[:, None] * projector / root_mass[None, :]
    assert np.linalg.norm(weighted - weighted.T) / np.linalg.norm(weighted) < 2.0e-13
    assert abs(np.linalg.norm(weighted, 2) - 1.0) < 2.0e-12


def test_resolved_affine_quadratic_green_compatibility() -> None:
    disc = affine.build_discretization(12)
    case = _representative_step()
    terminal = case["terminal_geometry"]
    action = affine.transport_action_sparse(
        disc.G0, terminal["inverse"], terminal["J"]
    )
    audit = affine.green_compatibility_audit(
        disc,
        case["D_q_terminal"],
        action,
        terminal,
        case["position"],
    )
    assert audit["full_space_relative_frobenius"] > 1.0e-2
    assert (
        audit["restricted_affine_quadratic"]["riesz_normalized_supremum"]
        < 2.0e-13
    )


def test_placed_action_stage_consistency() -> None:
    audit = affine.placed_action_stage_consistency(layers=12)
    assert audit["passed"]
    assert audit["rates"]["0.5"]["action_stage_error_rms"] > 1.8
    assert 0.8 < audit["rates"]["0.0"]["action_stage_error_rms"] < 1.25
    assert 0.8 < audit["rates"]["1.0"]["action_stage_error_rms"] < 1.25
    assert audit["maximum_spatial_action_reproduction_error_rms"] < 1.0e-11
    assert audit["rates"]["0.5"]["pressure_step_average_error_rms"] > 1.8
    assert 0.8 < audit["rates"]["0.0"]["pressure_step_average_error_rms"] < 1.25
    assert 0.8 < audit["rates"]["1.0"]["pressure_step_average_error_rms"] < 1.25


def test_deterministic_repeatability() -> None:
    first = _representative_step()
    second = _representative_step()
    assert np.array_equal(first["position"], second["position"])
    assert np.array_equal(first["velocity"], second["velocity"])
    assert np.array_equal(first["pressure"], second["pressure"])
