"""
Three-branch CCOP / matched-ISPH / SISPH-paper comparison harness.

Purpose
-------
This file is a controlled comparison refactor of the user's nonlinear CCOP
oscillating-drop code.  The three branches share the same particles, kernel,
neighbor graph, free-surface pressure mask, body-force predictor, time step,
and diagnostics as far as the comparison permits.

Branches
--------
1) --method ccop
   Nonlinear terminal-configuration closure

       R(p) = D[x_T(p)] (u* - dt G[x_theta(p)] p) = 0,

   with pressure-induced terminal position

       x_T = x* - c_x dt^2 G[x_theta] p.

   The full p -> x_theta -> G -> x_T -> D dependence is differentiated by
   JAX JVP and solved by Newton-Krylov with continuation.

2) --method matched-isph
   Same *common* SPH D/G routines as CCOP, but with frozen predictor geometry

       D[x*] (u* - dt G[x*] p) = 0.

   This is the mechanism-control baseline.  It is linear because x* is frozen.

3) --method sisph-paper
   A paper-stencil SISPH branch under the same outer harness.  It reproduces
   the core discretizations in Muta--Ramachandran--Negi's public SISPH code:
   summation density, the velocity-divergence RHS, the Brookshaw-type PPE
   coefficient, and the pressure-difference gradient.  It is intentionally
   *not* a byte-for-byte PySPH driver replica: predictor, particle set, and the
   default pressure-free boundary mask are kept common for a controlled test.

Important comparison contract
-----------------------------
* CCOP and matched-ISPH use exactly the same spatial SPH D and G; by default a moving first-order moment correction is applied to both.
* The default kernel is the 2-D PySPH-style QuinticSpline (support 3h).
* The neighbor topology is rebuilt at the beginning of every physical
  substep and then frozen during a pressure solve.  Kernel distances/gradients
  are re-evaluated at the requested configuration.
* The default paper-operator mode is ``shared``: all three branches use the
  same particles, fixed V0, rho0, D_h and G_h.  The SISPH branch differs only
  by replacing the composed D_h G_h pressure operator with a Brookshaw-type
  PPE stencil.  ``--paper-operator-mode original`` restores raw paper D/G and
  summation density for an external-method reproduction.
* Diagnostics separate branch-native and common operators, predictor and
  terminal configurations, and frozen versus independently rebuilt terminal
  neighbor graphs.

This is research comparison code.  Run refinement/tolerance audits before
using numerical differences as publication evidence.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csc_matrix, eye as sparse_eye
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, lsmr, splu

import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Utilities and geometry
# -----------------------------------------------------------------------------

def parse_float_list(s: str) -> List[float]:
    if s is None or s.strip() == "":
        return []
    return [float(x) for x in s.split(',') if x.strip()]


def generate_disk(radius: float, dx: float) -> np.ndarray:
    xs = np.arange(-radius, radius + 0.5 * dx, dx)
    pts = []
    for x in xs:
        for y in xs:
            if x * x + y * y <= radius * radius + 1e-12:
                pts.append((x, y))
    pts = np.asarray(pts, dtype=np.float64)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def generate_ring_disk(radius: float, dx: float, phase_jitter: bool = False,
                       seed: int = 0, add_bbox_points: bool = True,
                       n_theta_outer: int = 0, keep_center: bool = True,
                       tol: float = 1e-10):
    """Concentric-ring disk with an explicit material free-surface ring."""
    rng = np.random.default_rng(seed)
    dr = max(float(dx), 1e-12)

    xs_in, ys_in, ring_id_in = [], [], []
    if keep_center:
        xs_in.append(0.0)
        ys_in.append(0.0)
        ring_id_in.append(0)

    n_rings = int(np.floor(radius / dr))
    for k in range(1, n_rings + 1):
        r = k * dr
        if r >= radius:
            break
        n_theta = max(int(np.round(2.0 * np.pi * r / dr)), 6)
        if n_theta % 2 != 0:
            n_theta += 1
        offset = (rng.random() * 2.0 * np.pi) if phase_jitter else (
            0.5 * (k % 2) * 2.0 * np.pi / n_theta
        )
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + offset
        xs_in.extend(r * np.cos(theta))
        ys_in.extend(r * np.sin(theta))
        ring_id_in.extend([k] * n_theta)

    if xs_in:
        xy_in = np.column_stack([xs_in, ys_in]).astype(np.float64)
    else:
        xy_in = np.zeros((0, 2), dtype=np.float64)
    ring_id_in = np.asarray(ring_id_in, dtype=np.int64)

    if len(xy_in) > 0:
        xy_q = np.round(xy_in / tol) * tol
        _, unique_idx = np.unique(xy_q, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        xy_in = xy_q[unique_idx]
        ring_id_in = ring_id_in[unique_idx]

    if n_theta_outer is None or int(n_theta_outer) <= 0:
        n_theta_outer = max(int(np.round(2.0 * np.pi * radius / dr)), 32)
        if n_theta_outer % 2 != 0:
            n_theta_outer += 1
    else:
        n_theta_outer = int(n_theta_outer)

    offset = (rng.random() * 2.0 * np.pi) if phase_jitter else 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta_outer, endpoint=False) + offset
    xy_out = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    rr = np.linalg.norm(xy_out, axis=1)
    xy_out *= (radius / rr)[:, None]
    ring_id_out = np.full(n_theta_outer, 10**9, dtype=np.int64)

    extras = []
    if add_bbox_points:
        extras = [[-radius, 0.0], [radius, 0.0], [0.0, -radius], [0.0, radius]]
    xy_extra = (np.asarray(extras, dtype=np.float64).reshape((-1, 2))
                if extras else np.zeros((0, 2), dtype=np.float64))
    ring_id_extra = -np.ones(len(xy_extra), dtype=np.int64)

    start_out = len(xy_in)
    X = np.vstack([xy_in, xy_out, xy_extra]).astype(np.float64)
    ring_id = np.concatenate([ring_id_in, ring_id_out, ring_id_extra])
    outer_idx = np.arange(start_out, start_out + len(xy_out), dtype=np.int32)
    if len(xy_extra) > 0:
        outer_idx = np.concatenate([
            outer_idx,
            np.arange(start_out + len(xy_out), len(X), dtype=np.int32)
        ])

    rr_out = np.linalg.norm(X[outer_idx], axis=1)
    print(
        f'[ring-init] points={X.shape}, dr={dr:.8e}, boundary/free={len(outer_idx)}, '
        f'outer r=[{rr_out.min():.16e},{rr_out.max():.16e}], bbox={bool(add_bbox_points)}'
    )
    return X, outer_idx.astype(np.int32), ring_id


# -----------------------------------------------------------------------------
# SPH kernels: common spatial backbone
# -----------------------------------------------------------------------------

def kernel_support(kernel: str) -> float:
    return 2.0 if kernel == 'cubic' else 3.0


def kernel_w_np(r: np.ndarray | float, h: float, kernel: str) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    q = r / h
    if kernel == 'cubic':
        sigma = 10.0 / (7.0 * math.pi * h * h)
        out = np.zeros_like(q)
        m0 = q < 1.0
        m1 = (q >= 1.0) & (q < 2.0)
        out[m0] = sigma * (1.0 - 1.5 * q[m0]**2 + 0.75 * q[m0]**3)
        out[m1] = sigma * 0.25 * (2.0 - q[m1])**3
        return out

    sigma = 7.0 / (478.0 * math.pi * h * h)
    out = np.zeros_like(q)
    m0 = q < 1.0
    m1 = (q >= 1.0) & (q < 2.0)
    m2 = (q >= 2.0) & (q < 3.0)
    out[m0] = sigma * (
        (3.0 - q[m0])**5 - 6.0 * (2.0 - q[m0])**5
        + 15.0 * (1.0 - q[m0])**5
    )
    out[m1] = sigma * ((3.0 - q[m1])**5 - 6.0 * (2.0 - q[m1])**5)
    out[m2] = sigma * (3.0 - q[m2])**5
    return out


def build_step_graph(x: np.ndarray, h: float, kernel: str,
                     graph_skin: float = 0.25) -> Dict[str, np.ndarray]:
    """Build topology at substep start; topology is frozen during pressure solve."""
    N = len(x)
    radius = (kernel_support(kernel) + graph_skin) * h
    tree = cKDTree(x)
    lists = tree.query_ball_point(x, radius)
    cleaned, maxn = [], 0
    for i, li in enumerate(lists):
        cur = sorted(int(j) for j in li if j != i)
        cleaned.append(cur)
        maxn = max(maxn, len(cur))
    maxn = max(maxn, 1)

    neigh = np.zeros((N, maxn), dtype=np.int32)
    mask = np.zeros((N, maxn), dtype=np.float64)
    for i, cur in enumerate(cleaned):
        m = len(cur)
        if m:
            neigh[i, :m] = cur
            mask[i, :m] = 1.0
        if m < maxn:
            neigh[i, m:] = i
    return {'neigh': neigh, 'mask': mask, 'maxn': np.array([maxn], dtype=np.int32)}


def reference_center_pressure(t: float, R: float, delta0: float,
                              Omega: float, rho0: float) -> float:
    if t == 0.0:
        return 0.5 * rho0 * R * R * (delta0 * delta0 + Omega * Omega)

    def rhs(_t, y):
        a, delta = y
        rr = R ** 4 / a ** 4
        dd = ((rr - 1.0) / (rr + 1.0)) * (delta * delta + Omega * Omega)
        return [delta * a, dd]

    sol = solve_ivp(rhs, (0.0, t), [R, delta0], rtol=1e-10,
                    atol=1e-12, method='DOP853')
    a = sol.y[0, -1]
    delta = sol.y[1, -1]
    rr = R ** 4 / a ** 4
    delta_dot = ((rr - 1.0) / (rr + 1.0)) * (delta * delta + Omega * Omega)
    return 0.5 * rho0 * a * a * (delta_dot + delta * delta + Omega * Omega)


@dataclass
class StepResult:
    x: np.ndarray
    u: np.ndarray
    p_full: np.ndarray
    diag: Dict[str, float]


class StepSolveFailure(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Three-branch solver
# -----------------------------------------------------------------------------

class ThreeBranchISPH:
    def __init__(self, args):
        self.args = args
        jax.config.update('jax_enable_x64', bool(args.fp64))
        if args.backend != 'auto':
            jax.config.update('jax_platform_name', args.backend)
        self.dtype = jnp.float64 if args.fp64 else jnp.float32

        self.method = args.method
        self.kernel = args.kernel
        self.paper_operator_mode = args.paper_operator_mode
        self.R = args.radius
        self.dx = args.dx if args.dx > 0 else (2.0 * args.radius / args.resolution)
        self.h = args.h_factor * self.dx
        self.V0 = self.dx * self.dx
        self.rho0 = args.rho0
        self.m0 = self.rho0 * self.V0
        self.dt_nominal = args.dt
        self.dt = args.dt
        self.omega2 = args.omega ** 2 if args.omega2 < 0 else args.omega2

        if args.init_scheme == 'ring':
            X, material_free_idx, ring_id = generate_ring_disk(
                args.radius, self.dx, phase_jitter=bool(args.phase_jitter),
                seed=int(args.seed), add_bbox_points=bool(args.add_bbox_points),
                n_theta_outer=int(args.n_theta_outer), keep_center=True,
            )
            self.material_free_idx_np = material_free_idx.astype(np.int32)
            self.ring_id_np = ring_id
        else:
            X = generate_disk(args.radius, self.dx)
            self.material_free_idx_np = np.array([], dtype=np.int32)
            self.ring_id_np = np.zeros(len(X), dtype=np.int64)

        self.X_np = X
        self.N = X.shape[0]
        self.center_idx = int(np.argmin(np.sum(X * X, axis=1)))

        r = np.linalg.norm(X, axis=1)
        if args.free_surface_mode == 'material':
            if len(self.material_free_idx_np) == 0:
                raise ValueError('--free-surface-mode material requires --init-scheme ring')
            free = np.zeros(self.N, dtype=bool)
            free[self.material_free_idx_np] = True
        elif args.free_surface_mode == 'radial':
            free = r >= (args.radius - args.free_surface_width * self.dx)
        elif args.free_surface_mode == 'none':
            free = np.zeros(self.N, dtype=bool)
        else:
            raise ValueError('unknown free_surface_mode')

        # If no Dirichlet pressure boundary exists, pin one gauge pressure.
        self.gauge_idx = -1
        if not np.any(free):
            self.gauge_idx = self.center_idx if args.gauge_index < 0 else int(args.gauge_index)
            free[self.gauge_idx] = True
            print(f'[pressure-gauge] no free p=0 boundary: pin p[{self.gauge_idx}]=0')

        self.free_mask_np = free
        self.unknown_idx_np = np.where(~free)[0].astype(np.int32)
        self.free_idx_np = np.where(free)[0].astype(np.int32)
        self.n_unknown = len(self.unknown_idx_np)
        self.unknown_idx = jnp.asarray(self.unknown_idx_np)

        self.beta_grid = parse_float_list(args.beta_grid)
        if not self.beta_grid:
            self.beta_grid = [1.0, 0.5, 0.25, 0.125, 0.0]

        # Step graph is set before each substep solve.
        self.neigh_np = None
        self.mask_np = None
        self.neigh = None
        self.mask = None

        print('=== CCOP / matched-ISPH / SISPH-paper three-branch harness ===')
        print('method:', self.method)
        print('JAX devices:', jax.devices())
        print('default backend:', jax.default_backend())
        print(f'N={self.N}, unknown={self.n_unknown}, p0-boundary/gauge={len(self.free_idx_np)}')
        print(f'R={self.R}, dx={self.dx:.8e}, h={self.h:.8e}, h/dx={self.h/self.dx:.3f}')
        print(f'kernel={self.kernel}, support={kernel_support(self.kernel):.1f}h, graph_skin={args.graph_skin:.2f}h')
        print(f'common D/G operator={args.common_operator}, correction_reg={args.common_correction_reg:.1e}')
        print(f'dt={self.dt:.8e}, rho0={self.rho0}, common V0={self.V0:.8e}')
        print(f'predictor={args.predictor}, free_surface_mode={args.free_surface_mode}')
        if self.method == 'ccop':
            print(f'CCOP placement={args.placement}, theta={args.theta}, position_correction={args.position_correction}')
        elif self.method == 'matched-isph':
            print('matched-ISPH: same common D/G as CCOP; both frozen at x*')
        else:
            if self.paper_operator_mode == 'shared':
                print('sisph-paper(shared): common D/G, fixed V0/rho0, Brookshaw-type PPE only')
                print('CONTROLLED COMPARISON: source and pressure correction use exactly the common D/G.')
            else:
                print('sisph-paper(original): summation/raw paper D/G + Brookshaw PPE')
                print('EXTERNAL BASELINE: not discretely identical to CCOP/matched-ISPH.')
            print('NOTE: shared predictor/boundary harness; this is not a byte-for-byte PySPH driver replica.')

    # ------------------------------------------------------------------
    # Common state / graph / kernel operators
    # ------------------------------------------------------------------

    def effective_pressure_theta(self) -> float:
        if self.args.pressure_time_theta >= 0.0:
            return float(self.args.pressure_time_theta)
        if self.method != 'ccop':
            return 1.0
        if self.args.placement == 'midpoint':
            return 0.5
        if self.args.placement == 'current':
            return 0.0
        if self.args.placement in ('terminal', 'star'):
            return 1.0
        return float(self.args.theta)

    def initial_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.X_np.copy()
        u = np.zeros_like(x)
        u[:, 0] = self.args.delta0 * x[:, 0]
        u[:, 1] = -self.args.delta0 * x[:, 1]
        p = np.zeros(self.N, dtype=np.float64)
        return x, u, p

    def pack_unknown(self, p_full: np.ndarray) -> np.ndarray:
        return p_full[self.unknown_idx_np].astype(np.float64)

    def _unpack_jax(self, p_u: jnp.ndarray,
                    unknown_idx: jnp.ndarray | None = None) -> jnp.ndarray:
        idx = self.unknown_idx if unknown_idx is None else unknown_idx
        p = jnp.zeros((self.N,), dtype=self.dtype)
        return p.at[idx].set(p_u, unique_indices=True)

    def _prepare_step_graph(self, x_n_np: np.ndarray):
        g = build_step_graph(x_n_np, self.h, self.kernel, self.args.graph_skin)
        self.neigh_np = g['neigh']
        self.mask_np = g['mask']
        self.neigh = jnp.asarray(self.neigh_np)
        self.mask = jnp.asarray(self.mask_np, dtype=self.dtype)
        counts = self.mask_np.sum(axis=1)
        if np.min(counts) < self.args.min_neighbors_warn:
            print(
                f'[graph-warning] neighbor count min/mean/max='
                f'{counts.min():.0f}/{counts.mean():.1f}/{counts.max():.0f}'
            )

    def make_predictor(self, x_n_np: np.ndarray, u_n_np: np.ndarray):
        x_n = jnp.asarray(x_n_np, dtype=self.dtype)
        u_n = jnp.asarray(u_n_np, dtype=self.dtype)

        if self.args.predictor == 'kinematic':
            a_ext = -self.omega2 * x_n
            u_star = u_n + self.dt * a_ext
            x_star = x_n + self.dt * u_n + 0.5 * self.dt * self.dt * a_ext
        elif self.args.predictor == 'semi-implicit':
            a_ext = -self.omega2 * x_n
            u_star = u_n + self.dt * a_ext
            x_star = x_n + self.dt * u_star
        elif self.args.predictor == 'implicit-midpoint-force':
            denom = 1.0 + 0.25 * self.omega2 * self.dt * self.dt
            u_mid = (u_n - 0.5 * self.omega2 * self.dt * x_n) / denom
            x_star = x_n + self.dt * u_mid
            u_star = 2.0 * u_mid - u_n
        else:
            raise ValueError('unknown predictor')
        return x_star, u_star

    def _kernel_w_jax(self, r: jnp.ndarray) -> jnp.ndarray:
        q = r / self.h
        if self.kernel == 'cubic':
            sigma = 10.0 / (7.0 * math.pi * self.h * self.h)
            f0 = 1.0 - 1.5 * q*q + 0.75 * q*q*q
            f1 = 0.25 * (2.0 - q)**3
            return sigma * jnp.where(q < 1.0, f0,
                                     jnp.where(q < 2.0, f1, 0.0))

        sigma = 7.0 / (478.0 * math.pi * self.h * self.h)
        f0 = (3.0-q)**5 - 6.0*(2.0-q)**5 + 15.0*(1.0-q)**5
        f1 = (3.0-q)**5 - 6.0*(2.0-q)**5
        f2 = (3.0-q)**5
        return sigma * jnp.where(q < 1.0, f0,
                                 jnp.where(q < 2.0, f1,
                                           jnp.where(q < 3.0, f2, 0.0)))

    def _pair_geometry_raw(self, x: jnp.ndarray):
        # PySPH convention: XIJ = x_i - x_j and DWIJ = grad_i W_ij.
        xj = x[self.neigh]
        xi = x[:, None, :]
        rij_vec = (xi - xj) * self.mask[:, :, None]
        r2 = jnp.sum(rij_vec * rij_vec, axis=2)
        # Smooth zero-distance regularization is essential for AD-JVP/VJP: padded
        # self entries have r=0, and differentiating sqrt(r2) at zero otherwise
        # produces NaNs even though those entries are masked.
        pair_eps = 1e-12 * self.h
        r = jnp.sqrt(jnp.maximum(r2, 0.0) + pair_eps * pair_eps)
        q = r / self.h

        if self.kernel == 'cubic':
            sigma = 10.0 / (7.0 * math.pi * self.h * self.h)
            d0 = -3.0*q + 2.25*q*q
            d1 = -0.75*(2.0-q)**2
            dfdq = jnp.where(q < 1.0, d0, jnp.where(q < 2.0, d1, 0.0))
        else:
            sigma = 7.0 / (478.0 * math.pi * self.h * self.h)
            d0 = -5.0*(3.0-q)**4 + 30.0*(2.0-q)**4 - 75.0*(1.0-q)**4
            d1 = -5.0*(3.0-q)**4 + 30.0*(2.0-q)**4
            d2 = -5.0*(3.0-q)**4
            dfdq = jnp.where(q < 1.0, d0,
                              jnp.where(q < 2.0, d1,
                                        jnp.where(q < 3.0, d2, 0.0)))

        dWdr = sigma * dfdq / self.h
        invr = 1.0 / r
        grad = dWdr[:, :, None] * rij_vec * invr[:, :, None]
        grad = grad * self.mask[:, :, None]
        return rij_vec, r, r2, grad

    def _pair_geometry_common(self, x: jnp.ndarray):
        rij_vec, r, r2, grad = self._pair_geometry_raw(x)
        if self.args.common_operator == 'raw':
            return rij_vec, r, r2, grad

        # Moving-configuration first-order moment correction.  With
        # dX_ij=x_j-x_i=-XIJ, enforce
        #     V0 sum_j dX_ij \otimes gradWcorr_ij = I.
        dX = -rij_vec
        A = self.V0 * jnp.einsum('nka,nkb->nab', dX, grad)
        eye = jnp.eye(2, dtype=self.dtype)[None, :, :]
        Areg = A + self.args.common_correction_reg * eye
        invAT = jnp.swapaxes(jnp.linalg.inv(Areg), 1, 2)
        grad_corr = jnp.einsum('ncb,nkb->nkc', invAT, grad)
        grad_corr = grad_corr * self.mask[:, :, None]
        return rij_vec, r, r2, grad_corr

    def _D_common(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        _, _, _, grad = self._pair_geometry_common(x)
        uj = u[self.neigh]
        ui = u[:, None, :]
        du = (uj - ui) * self.mask[:, :, None]
        return self.V0 * jnp.einsum('nka,nka->n', du, grad)

    def _G_common(self, x: jnp.ndarray, p_full: jnp.ndarray) -> jnp.ndarray:
        _, _, _, grad = self._pair_geometry_common(x)
        pj = p_full[self.neigh]
        pi = p_full[:, None]
        dp = (pj - pi) * self.mask
        return (self.V0 / self.rho0) * jnp.einsum('nk,nka->na', dp, grad)

    def _density_paper(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.paper_operator_mode == 'shared' or self.args.paper_density == 'rho0':
            return jnp.full((self.N,), self.rho0, dtype=self.dtype)
        _, r, _, _ = self._pair_geometry_raw(x)
        wij = self._kernel_w_jax(r) * self.mask
        self_w = self._kernel_w_jax(jnp.zeros((self.N,), dtype=self.dtype))
        rho = self.m0 * self_w + self.m0 * jnp.sum(wij, axis=1)
        return jnp.maximum(rho, self.args.paper_density_floor * self.rho0)

    def _D_paper(self, x: jnp.ndarray, u: jnp.ndarray,
                 rho: jnp.ndarray) -> jnp.ndarray:
        if self.paper_operator_mode == 'shared':
            return self._D_common(x, u)
        _, _, _, grad = self._pair_geometry_raw(x)
        uj = u[self.neigh]
        ui = u[:, None, :]
        du = (uj - ui) * self.mask[:, :, None]
        Vj = (self.m0 / rho[self.neigh]) * self.mask
        return jnp.einsum('nk,nka,nka->n', Vj, du, grad)

    def _G_paper(self, x: jnp.ndarray, p_full: jnp.ndarray,
                 rho: jnp.ndarray) -> jnp.ndarray:
        if self.paper_operator_mode == 'shared':
            return self._G_common(x, p_full)
        _, _, _, grad = self._pair_geometry_raw(x)
        pj = p_full[self.neigh]
        pi = p_full[:, None]
        dp = (pj - pi) * self.mask
        Vj = (self.m0 / rho[self.neigh]) * self.mask
        return jnp.einsum('nk,nk,nka->na', Vj, dp, grad) / rho[:, None]

    def _A_paper_full(self, x: jnp.ndarray, p_full: jnp.ndarray,
                      rho: jnp.ndarray) -> jnp.ndarray:
        if self.paper_operator_mode == 'shared':
            rij_vec, _, r2, grad = self._pair_geometry_common(x)
            rho = jnp.full((self.N,), self.rho0, dtype=self.dtype)
        else:
            rij_vec, _, r2, grad = self._pair_geometry_raw(x)
        xdotdwij = jnp.einsum('nka,nka->nk', rij_vec, grad)
        rhoi = rho[:, None]
        rhoj = rho[self.neigh]
        fac = (
            4.0 * self.m0 / (rhoi * (rhoi + rhoj))
            * xdotdwij / (r2 + self.args.paper_eps)
        ) * self.mask
        pj = p_full[self.neigh]
        pi = p_full[:, None]
        # Exact algebra of diag=sum(fac), offdiag=-fac.
        return jnp.sum(fac * (pi - pj), axis=1)

    @staticmethod
    def _stats(a: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(a, dtype=np.float64)
        return float(np.max(np.abs(a))), float(np.sqrt(np.mean(a*a) + 1e-30))

    def _graph_snapshot(self):
        return (self.neigh_np, self.mask_np, self.neigh, self.mask)

    def _restore_graph(self, snap):
        self.neigh_np, self.mask_np, self.neigh, self.mask = snap

    def _set_graph_dict(self, g: Dict[str, np.ndarray]):
        self.neigh_np = g['neigh']
        self.mask_np = g['mask']
        self.neigh = jnp.asarray(self.neigh_np)
        self.mask = jnp.asarray(self.mask_np, dtype=self.dtype)

    @staticmethod
    def _edge_set(neigh: np.ndarray, mask: np.ndarray):
        return {
            (int(i), int(neigh[i, k]))
            for i in range(neigh.shape[0])
            for k in range(neigh.shape[1]) if mask[i, k] != 0.0
        }

    def _terminal_rebuilt_diagnostics(self, xT_np: np.ndarray,
                                      uT_np: np.ndarray) -> Dict[str, float]:
        """Independently rebuild the compact-support graph at accepted x_T.

        The nonlinear/linear solve keeps the step-start topology frozen for
        differentiability.  This routine checks that no pair entering kernel
        support at x_T was omitted and evaluates terminal divergence on a fresh
        graph, so a small terminal value cannot be an artifact of reusing x*.
        """
        snap = self._graph_snapshot()
        old_edges = self._edge_set(self.neigh_np, self.mask_np)
        fresh = build_step_graph(xT_np, self.h, self.kernel, graph_skin=0.0)
        fresh_edges = self._edge_set(fresh['neigh'], fresh['mask'])
        missing = fresh_edges.difference(old_edges)
        try:
            self._set_graph_dict(fresh)
            xT = jnp.asarray(xT_np, dtype=self.dtype)
            uT = jnp.asarray(uT_np, dtype=self.dtype)
            dc_full = np.asarray(self._D_common(xT, uT), dtype=np.float64)
            if self.method == 'sisph-paper':
                rhoT = self._density_paper(xT)
                dn_full = np.asarray(self._D_paper(xT, uT, rhoT), dtype=np.float64)
                rho_np = np.asarray(rhoT, dtype=np.float64)
            else:
                dn_full = dc_full.copy()
                rho_np = np.full(self.N, self.rho0)
        finally:
            self._restore_graph(snap)
        dc = dc_full[self.unknown_idx_np]
        dn = dn_full[self.unknown_idx_np]
        dc_free = dc_full[self.free_idx_np]
        dn_free = dn_full[self.free_idx_np]
        mc, rc = self._stats(dc)
        mn, rn = self._stats(dn)
        mca, rca = self._stats(dc_full)
        mna, rna = self._stats(dn_full)
        mcf, rcf = self._stats(dc_free)
        mnf, rnf = self._stats(dn_free)
        return {
            'max_D_common_terminal_rebuilt': mc,
            'rms_D_common_terminal_rebuilt': rc,
            'max_D_native_terminal_rebuilt': mn,
            'rms_D_native_terminal_rebuilt': rn,
            'max_D_common_terminal_all_rebuilt': mca,
            'rms_D_common_terminal_all_rebuilt': rca,
            'max_D_native_terminal_all_rebuilt': mna,
            'rms_D_native_terminal_all_rebuilt': rna,
            'max_D_common_terminal_free_rebuilt': mcf,
            'rms_D_common_terminal_free_rebuilt': rcf,
            'max_D_native_terminal_free_rebuilt': mnf,
            'rms_D_native_terminal_free_rebuilt': rnf,
            'terminal_missing_directed_edges': float(len(missing)),
            'terminal_fresh_directed_edges': float(len(fresh_edges)),
            'rho_terminal_min': float(np.min(rho_np)),
            'rho_terminal_max': float(np.max(rho_np)),
        }

    def _energy_budget(self, x_n_np: np.ndarray, u_n_np: np.ndarray,
                       xT_np: np.ndarray, uT_np: np.ndarray) -> Dict[str, float]:
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = np.asarray(x_star, dtype=np.float64)
        u_star = np.asarray(u_star, dtype=np.float64)
        Kn, Vn, En = self.compute_energy(np.asarray(x_n_np), np.asarray(u_n_np))
        Ks, Vs, Es = self.compute_energy(x_star, u_star)
        KT, VT, ET = self.compute_energy(xT_np, uT_np)
        g = (u_star - uT_np) / self.dt
        ubar = 0.5 * (u_star + uT_np)
        pressure_kinetic_work = -self.dt * self.m0 * float(np.sum(ubar * g))
        return {
            'energy_n': En, 'energy_star': Es, 'energy_T': ET,
            'kinetic_n': Kn, 'kinetic_star': Ks, 'kinetic_T': KT,
            'potential_n': Vn, 'potential_star': Vs, 'potential_T': VT,
            'predictor_energy_rel': (Es-En) / max(abs(En), 1e-30),
            'pressure_stage_energy_rel': (ET-Es) / max(abs(En), 1e-30),
            'energy_step_rel': (ET-En) / max(abs(En), 1e-30),
            'pressure_kinetic_work': pressure_kinetic_work,
            'pressure_kinetic_change': KT-Ks,
            'pressure_potential_change': VT-Vs,
            'pressure_work_identity_resid': (KT-Ks)-pressure_kinetic_work,
        }

    def _augment_step_diagnostics(self, result: StepResult,
                                  x_n_np: np.ndarray,
                                  u_n_np: np.ndarray) -> StepResult:
        """Add unambiguous native/common and frozen/rebuilt diagnostics."""
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = jnp.asarray(x_star, dtype=self.dtype)
        u_star = jnp.asarray(u_star, dtype=self.dtype)
        uT = jnp.asarray(result.u, dtype=self.dtype)
        xT = jnp.asarray(result.x, dtype=self.dtype)

        dc0 = np.asarray(self._D_common(x_star, u_star), dtype=np.float64)[self.unknown_idx_np]
        dcs = np.asarray(self._D_common(x_star, uT), dtype=np.float64)[self.unknown_idx_np]
        dcT = np.asarray(self._D_common(xT, uT), dtype=np.float64)[self.unknown_idx_np]

        if self.method == 'sisph-paper':
            rho_star = self._density_paper(x_star)
            dn0 = np.asarray(self._D_paper(x_star, u_star, rho_star), dtype=np.float64)[self.unknown_idx_np]
            dns = np.asarray(self._D_paper(x_star, uT, rho_star), dtype=np.float64)[self.unknown_idx_np]
            native_repr = dns
            rho_np = np.asarray(rho_star, dtype=np.float64)
        else:
            dn0 = dc0.copy()
            dns = dcs.copy()
            native_repr = dcT if self.method == 'ccop' else dcs
            rho_np = np.full(self.N, self.rho0)

        d = result.diag
        for prefix, arr in [
            ('D_common_star_before', dc0),
            ('D_common_star_after', dcs),
            ('D_common_terminal_frozen', dcT),
            ('D_native_star_before', dn0),
            ('D_native_star_after', dns),
            ('D_native_repr_after', native_repr),
        ]:
            mx, rr = self._stats(arr)
            d['max_' + prefix] = mx
            d['rms_' + prefix] = rr

        d.update(self._terminal_rebuilt_diagnostics(np.asarray(result.x), np.asarray(result.u)))
        d.update(self._energy_budget(x_n_np, u_n_np, np.asarray(result.x), np.asarray(result.u)))
        d['rho_star_min'] = float(np.min(rho_np))
        d['rho_star_max'] = float(np.max(rho_np))

        # Backward-compatible keys with now-explicit semantics.
        d['max_D_star'], d['rms_D_star'] = self._stats(dn0)
        d['max_D_repr'], d['rms_D_repr'] = self._stats(native_repr)
        d['max_D_terminal'] = d['max_D_common_terminal_rebuilt']
        d['rms_D_terminal'] = d['rms_D_common_terminal_rebuilt']
        d['rms_D_frozen_after'] = d['rms_D_common_star_after']
        return result

    def audit_discrete_parity(self, x_n_np: np.ndarray, u_n_np: np.ndarray):
        """Numerically verify the common/paper D and G identities at x*."""
        self._prepare_step_graph(x_n_np)
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = jnp.asarray(x_star, dtype=self.dtype)
        u_star = jnp.asarray(u_star, dtype=self.dtype)
        rho = self._density_paper(x_star)
        ptest_np = np.maximum(self.R*self.R - np.sum(np.asarray(x_star)**2, axis=1), 0.0)
        ptest_np[self.free_idx_np] = 0.0
        ptest = jnp.asarray(ptest_np, dtype=self.dtype)
        dc = np.asarray(self._D_common(x_star, u_star))[self.unknown_idx_np]
        dp = np.asarray(self._D_paper(x_star, u_star, rho))[self.unknown_idx_np]
        gc = np.asarray(self._G_common(x_star, ptest))
        gp = np.asarray(self._G_paper(x_star, ptest, rho))
        Ap = np.asarray(self._A_paper_full(x_star, ptest, rho))[self.unknown_idx_np]
        DGp = np.asarray(self._D_paper(x_star, self._G_paper(x_star, ptest, rho), rho))[self.unknown_idx_np]
        drel = np.linalg.norm(dc-dp) / max(np.linalg.norm(dc), 1e-30)
        grel = np.linalg.norm(gc-gp) / max(np.linalg.norm(gc), 1e-30)
        arel = np.linalg.norm(Ap-DGp) / max(np.linalg.norm(Ap), 1e-30)
        Bsp, Dsp, Gsp = self._assemble_frozen_common_sparse(x_star, return_operators=True)
        p_u_test = ptest_np[self.unknown_idx_np]
        sparse_DGp = np.asarray(Bsp @ p_u_test).ravel()
        sparse_rel = np.linalg.norm(sparse_DGp-DGp) / max(np.linalg.norm(DGp), 1e-30)
        adj = (self.m0 * Gsp + self.V0 * Dsp.T).tocsr()
        adj_norm = float(np.linalg.norm(adj.data)) if adj.nnz else 0.0
        g_norm = float(np.linalg.norm((self.m0 * Gsp).data)) if Gsp.nnz else 0.0
        dt_norm = float(np.linalg.norm((self.V0 * Dsp.T).data)) if Dsp.nnz else 0.0
        adj_rel = adj_norm / max(g_norm, dt_norm, 1e-30)
        print(
            f'[parity-audit] paper_mode={self.paper_operator_mode}, '
            f'||Dcommon-Dpaper||/||Dcommon||={drel:.3e}, '
            f'||Gcommon-Gpaper||/||Gcommon||={grel:.3e}, '
            f'||A_B-DG||/||A_B||={arel:.3e}, '
            f'||B_sparse-DG||/||DG||={sparse_rel:.3e}, '
            f'energy-adjoint-defect={adj_rel:.3e}'
        )
        return {
            'D_rel': drel, 'G_rel': grel, 'A_vs_DG_rel': arel,
            'sparse_B_vs_DG_rel': sparse_rel,
            'energy_adjoint_defect_rel': adj_rel,
        }

    def _assemble_frozen_common_sparse(self, x_star: jnp.ndarray, return_operators: bool = False):
        """Assemble the exact frozen composition B=D(x*)G(x*) sparsely.

        The entries are built from the *same* corrected/raw pair gradients used
        by ``_D_common`` and ``_G_common``.  This avoids a costly JAX-vmap dense
        basis assembly and makes the linear matched branch a reliable algebraic
        control rather than an iterative-solver experiment.
        """
        grad = np.asarray(self._pair_geometry_common(x_star)[3], dtype=np.float64)
        neigh = self.neigh_np
        mask = self.mask_np
        unknown_map = -np.ones(self.N, dtype=np.int64)
        unknown_map[self.unknown_idx_np] = np.arange(self.n_unknown, dtype=np.int64)

        # D: R^{2N} -> R^{n_unknown}.
        dr, dc, dv = [], [], []
        for row, i in enumerate(self.unknown_idx_np):
            active = np.flatnonzero(mask[i] != 0.0)
            if active.size == 0:
                continue
            js = neigh[i, active]
            gi = grad[i, active]
            gsum = np.sum(gi, axis=0)
            for a in range(2):
                vals = self.V0 * gi[:, a]
                dr.extend([row] * len(js)); dc.extend((2*js + a).tolist()); dv.extend(vals.tolist())
                dr.append(row); dc.append(2*int(i) + a); dv.append(float(-self.V0 * gsum[a]))
        D = coo_matrix((dv, (dr, dc)), shape=(self.n_unknown, 2*self.N)).tocsr()

        # G: R^{n_unknown} -> R^{2N}, with p=0 on the common free/gauge set.
        gr, gc, gv = [], [], []
        cG = self.V0 / self.rho0
        for i in range(self.N):
            active = np.flatnonzero(mask[i] != 0.0)
            if active.size == 0:
                continue
            js = neigh[i, active]
            gi = grad[i, active]
            gsum = np.sum(gi, axis=0)
            ci = int(unknown_map[i])
            cj = unknown_map[js]
            for a in range(2):
                row = 2*i + a
                valid = cj >= 0
                if np.any(valid):
                    gr.extend([row] * int(np.count_nonzero(valid)))
                    gc.extend(cj[valid].tolist())
                    gv.extend((cG * gi[valid, a]).tolist())
                if ci >= 0:
                    gr.append(row); gc.append(ci); gv.append(float(-cG * gsum[a]))
        G = coo_matrix((gv, (gr, gc)), shape=(2*self.N, self.n_unknown)).tocsr()
        B = (D @ G).tocsc()
        if return_operators:
            return B, D, G
        return B

    def _assemble_frozen_common_matrix(self, x_star: jnp.ndarray) -> np.ndarray:
        """Dense compatibility wrapper around the exact sparse assembly."""
        return self._assemble_frozen_common_sparse(x_star).toarray()

    def _factor_frozen_common_sparse(self, x_star: jnp.ndarray):
        """Sparse LU of D(x*)G(x*), with a tiny shift only on true failure."""
        B = self._assemble_frozen_common_sparse(x_star)
        shifted = False
        try:
            fac = splu(B)
        except RuntimeError:
            scale = max(1.0, float(np.max(np.abs(B).sum(axis=1))))
            shift = self.args.preconditioner_shift * scale
            fac = splu(B + shift * sparse_eye(self.n_unknown, format='csc'))
            shifted = True
        return B, fac, shifted

    def _solve_frozen_common_pressure(self, x_star: jnp.ndarray,
                                      u_star: jnp.ndarray,
                                      p_prev_full: np.ndarray,
                                      tol: float,
                                      maxiter: int):
        """Least-squares solve of the frozen common equation D G p = D u*/dt.

        Used both by matched-ISPH and as a solver-only CCOP warm start.  It does
        not change the CCOP residual/equation.
        """
        source_full = self._D_common(x_star, u_star)
        source = np.asarray(source_full[self.unknown_idx], dtype=np.float64)
        b = source / self.dt
        b_norm = max(1e-30, float(np.linalg.norm(b)))

        @jax.jit
        def B_apply(p_u):
            p_full = self._unpack_jax(p_u)
            g = self._G_common(x_star, p_full)
            return self._D_common(x_star, g)[self.unknown_idx]

        zero_p = jnp.zeros((self.n_unknown,), dtype=self.dtype)

        @jax.jit
        def BT_apply(y):
            return jax.linear_transpose(B_apply, zero_p)(y)[0]

        def matvec(v):
            return np.array(B_apply(jnp.asarray(v, dtype=self.dtype)),
                            dtype=np.float64, copy=True)

        def rmatvec(v):
            return np.array(BT_apply(jnp.asarray(v, dtype=self.dtype)),
                            dtype=np.float64, copy=True)

        Aop = LinearOperator((self.n_unknown, self.n_unknown), matvec=matvec,
                             rmatvec=rmatvec, dtype=np.float64)
        if self.args.pressure_init == 'previous':
            x0 = self.pack_unknown(p_prev_full)
        else:
            x0 = np.zeros(self.n_unknown, dtype=np.float64)

        if self.args.matched_linear_solver == 'sparse-lu':
            Bsp, fac, shifted = self._factor_frozen_common_sparse(x_star)
            p_u = fac.solve(b)
            info = 1 if shifted else 0
            itn = 1
            self._last_frozen_sparse = Bsp
            self._last_frozen_sparse_lu = fac
            self._last_frozen_sparse_shifted = shifted
        elif self.args.matched_linear_solver == 'dense':
            B = self._assemble_frozen_common_matrix(x_star)
            try:
                p_u = np.linalg.solve(B, b)
                info = 0
            except np.linalg.LinAlgError:
                shift = self.args.preconditioner_shift * max(1.0, float(np.linalg.norm(B, ord=np.inf)))
                p_u, *_ = np.linalg.lstsq(B + shift*np.eye(self.n_unknown), b, rcond=1e-12)
                info = 1
            itn = 1
            self._last_frozen_matrix = B
        elif self.args.matched_linear_solver == 'lsmr':
            sol = lsmr(Aop, b, damp=self.args.matched_tikhonov,
                       atol=tol, btol=tol, conlim=self.args.matched_conlim,
                       maxiter=maxiter, x0=x0)
            p_u = sol[0]
            info = int(sol[1])
            itn = int(sol[2])
        else:
            counter = {'n': 0}
            def cb(_): counter['n'] += 1
            p_u, ginfo = gmres(Aop, b, x0=x0, rtol=tol, atol=0.0,
                               restart=self.args.gmres_restart,
                               maxiter=maxiter, callback=cb,
                               callback_type='legacy')
            info = int(ginfo)
            itn = int(counter['n'])

        rel = float(np.linalg.norm(matvec(p_u) - b) / b_norm)
        return p_u, source_full, rel, info, itn

    # ------------------------------------------------------------------
    # CCOP nonlinear branch
    # ------------------------------------------------------------------

    def _ccop_action_position(self, x_n: jnp.ndarray, x_star: jnp.ndarray,
                              xT: jnp.ndarray) -> jnp.ndarray:
        if self.args.placement == 'star':
            return x_star
        if self.args.placement == 'current':
            return x_n
        if self.args.placement == 'terminal':
            return xT
        if self.args.placement == 'midpoint':
            return 0.5 * (x_n + xT)
        theta = self.args.theta
        return (1.0 - theta) * x_n + theta * xT

    def _compute_ccop_state(self, p_u: jnp.ndarray, x_n: jnp.ndarray,
                            u_star: jnp.ndarray, x_star: jnp.ndarray,
                            lam: float):
        p_full = self._unpack_jax(p_u)
        xT0 = x_star

        def body(_, state):
            xT, done = state
            x_theta = self._ccop_action_position(x_n, x_star, xT)
            g = self._G_common(x_theta, p_full)
            x_new = x_star - lam * self.args.position_correction * self.dt * self.dt * g
            err = jnp.max(jnp.sqrt(jnp.sum((x_new - xT)**2, axis=1)))
            newly_done = err <= self.args.geometry_tol_rel * self.dx
            x_next = jnp.where(done, xT, x_new)
            return (x_next, jnp.logical_or(done, newly_done))

        xT, _ = jax.lax.fori_loop(
            0, self.args.geometry_iters, body,
            (xT0, jnp.array(False))
        )
        x_theta = self._ccop_action_position(x_n, x_star, xT)
        g = self._G_common(x_theta, p_full)
        uT = u_star - lam * self.dt * g
        xT_consistent = x_star - lam * self.args.position_correction * self.dt * self.dt * g
        geom_res = jnp.max(jnp.sqrt(jnp.sum((xT - xT_consistent)**2, axis=1))) / self.dx
        # Return the pressure-consistent explicit expression as accepted terminal x.
        xT = xT_consistent
        return p_full, g, uT, xT, geom_res

    def make_ccop_functions(self, x_n_np: np.ndarray, u_n_np: np.ndarray):
        x_n = jnp.asarray(x_n_np, dtype=self.dtype)
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        source0_full = self._D_common(x_star, u_star)
        source0 = source0_full[self.unknown_idx]

        def residual(p_u: jnp.ndarray, lam: float) -> jnp.ndarray:
            _, _, uT, xT, _ = self._compute_ccop_state(
                p_u, x_n, u_star, x_star, lam
            )
            divT = self._D_common(xT, uT)
            return divT[self.unknown_idx] - (1.0 - lam) * source0

        def diagnostics(p_u: jnp.ndarray, lam: float):
            p_full, g, uT, xT, geom_res = self._compute_ccop_state(
                p_u, x_n, u_star, x_star, lam
            )
            dstar0 = source0_full[self.unknown_idx]
            drepr_full = self._D_common(xT, uT)
            drepr = drepr_full[self.unknown_idx]
            dterminal = drepr
            dstar_corrected = self._D_common(x_star, uT)[self.unknown_idx]
            gnorm = jnp.sqrt(jnp.sum(g*g, axis=1))
            disp = jnp.sqrt(jnp.sum((xT-x_star)**2, axis=1))
            return (
                jnp.max(jnp.abs(dstar0)), jnp.sqrt(jnp.mean(dstar0*dstar0)+1e-30),
                jnp.max(jnp.abs(drepr)), jnp.sqrt(jnp.mean(drepr*drepr)+1e-30),
                jnp.max(jnp.abs(dterminal)), jnp.sqrt(jnp.mean(dterminal*dterminal)+1e-30),
                jnp.sqrt(jnp.mean(dstar_corrected*dstar_corrected)+1e-30),
                jnp.max(jnp.abs(p_full)), jnp.max(gnorm), jnp.max(disp),
                p_full[self.center_idx], geom_res,
                xT, uT, p_full, g,
            )

        def residual_jvp(p_u: jnp.ndarray, v: jnp.ndarray, lam: float):
            _, tangent = jax.jvp(lambda pp: residual(pp, lam), (p_u,), (v,))
            return tangent

        def residual_vjp(p_u: jnp.ndarray, y: jnp.ndarray, lam: float):
            _, pullback = jax.vjp(lambda pp: residual(pp, lam), p_u)
            return pullback(y)[0]

        return (
            jax.jit(residual), jax.jit(diagnostics),
            jax.jit(residual_jvp), jax.jit(residual_vjp), np.asarray(source0)
        )

    def solve_ccop_one_step(self, step: int, x_n_np: np.ndarray,
                            u_n_np: np.ndarray, p_prev_full: np.ndarray) -> StepResult:
        self._prepare_step_graph(x_n_np)
        residual_jit, diag_jit, jvp_jit, vjp_jit, source0_np = self.make_ccop_functions(x_n_np, u_n_np)
        source_norm = max(1e-30, float(np.linalg.norm(source0_np)))
        self._last_frozen_matrix = None
        self._last_frozen_sparse = None
        self._last_frozen_sparse_lu = None
        self._last_frozen_sparse_shifted = False
        warm_info = 0
        warm_it = 0
        warm_rel = np.nan
        if self.args.ccop_warmstart == 'matched':
            x_star_ws, u_star_ws = self.make_predictor(x_n_np, u_n_np)
            p_u, _, warm_rel, warm_info, warm_it = self._solve_frozen_common_pressure(
                jnp.asarray(x_star_ws, dtype=self.dtype),
                jnp.asarray(u_star_ws, dtype=self.dtype),
                p_prev_full, self.args.warmstart_tol, self.args.warmstart_iters
            )
        elif self.args.ccop_warmstart == 'previous':
            p_u = self.pack_unknown(p_prev_full)
        else:
            p_u = np.zeros(self.n_unknown, dtype=np.float64)

        # Frozen-DG right/left preconditioner for the nonlinear JVP solve.
        # For R(p)=D_T(u*-lambda*dt*G_theta p), J is approximately
        # -lambda*dt*D(x*)G(x*).  The factorization changes only solver cost,
        # never the nonlinear residual or accepted solution.
        frozen_solve = None
        if self.args.ccop_preconditioner == 'frozen-dg':
            fac = getattr(self, '_last_frozen_sparse_lu', None)
            if fac is None:
                x_star_pre, _ = self.make_predictor(x_n_np, u_n_np)
                _, fac, _ = self._factor_frozen_common_sparse(
                    jnp.asarray(x_star_pre, dtype=self.dtype)
                )
            frozen_solve = fac.solve

        def res_np(pv, lv):
            return np.array(residual_jit(jnp.asarray(pv, dtype=self.dtype), float(lv)), dtype=np.float64, copy=True)

        def jvp_np(pv, vv, lv):
            return np.array(jvp_jit(
                jnp.asarray(pv, dtype=self.dtype),
                jnp.asarray(vv, dtype=self.dtype), float(lv)
            ), dtype=np.float64, copy=True)

        def vjp_np(pv, yy, lv):
            return np.array(vjp_jit(
                jnp.asarray(pv, dtype=self.dtype),
                jnp.asarray(yy, dtype=self.dtype), float(lv)
            ), dtype=np.float64, copy=True)

        t0 = time.time()
        lam = 0.0
        lam_step = self.args.lambda_step
        reductions = 0
        accepted_newton = 0
        last_gmres_info, last_gmres_it = 0, 0
        last_beta = 0.0
        last_rel = 1.0

        # p=0 solves lambda=0 by construction; continue to lambda=1.
        while lam < 1.0 - self.args.lambda_complete_tol:
            # If the current p already solves the full problem, snap to lambda=1.
            r1 = res_np(p_u, 1.0)
            rel1 = float(np.linalg.norm(r1) / source_norm)
            if rel1 <= self.args.final_accept_rel * (1.0 + self.args.accept_rel_slack):
                lam = 1.0
                last_rel = rel1
                break

            target = min(1.0, lam + lam_step)
            p_stage = p_u.copy()
            success = False

            for _nit in range(self.args.newton_iters):
                r0 = res_np(p_stage, target)
                rel0 = float(np.linalg.norm(r0) / source_norm)
                accept_tol = (self.args.final_accept_rel
                              if target >= 1.0 - self.args.lambda_complete_tol
                              else self.args.lambda_accept_rel)
                last_rel = rel0
                if rel0 <= accept_tol * (1.0 + self.args.accept_rel_slack):
                    success = True
                    break

                gmres_counter = {'n': 0}

                if self.args.jvp_mode == 'autodiff':
                    def matvec(v):
                        if np.linalg.norm(v) < 1e-300:
                            return np.zeros_like(v)
                        return jvp_np(p_stage, v, target) + self.args.newton_damp * v
                    def rmatvec(y):
                        if np.linalg.norm(y) < 1e-300:
                            return np.zeros_like(y)
                        return vjp_np(p_stage, y, target) + self.args.newton_damp * y
                else:
                    pnorm = np.linalg.norm(p_stage)
                    eps_base = self.args.jvp_eps * (1.0 + pnorm)
                    def matvec(v):
                        nv = np.linalg.norm(v)
                        if nv < 1e-300:
                            return np.zeros_like(v)
                        eps = eps_base / nv
                        return ((res_np(p_stage + eps*v, target)-r0)/eps
                                + self.args.newton_damp*v)
                    # FD mode has no cheap exact transpose; use autodiff VJP of
                    # the true residual for the least-squares transpose.
                    def rmatvec(y):
                        return vjp_np(p_stage, y, target) + self.args.newton_damp*y

                Aop = LinearOperator((self.n_unknown, self.n_unknown), matvec=matvec,
                                     rmatvec=rmatvec, dtype=np.float64)

                if self.args.newton_linear_solver == 'lsmr':
                    sol = lsmr(Aop, -r0, damp=self.args.newton_lsmr_damp,
                               atol=self.args.gmres_tol, btol=self.args.gmres_tol,
                               conlim=self.args.matched_conlim,
                               maxiter=self.args.gmres_iters)
                    delta = sol[0]
                    info = int(sol[1])
                    gmres_counter['n'] = int(sol[2])
                else:
                    def cb(_):
                        gmres_counter['n'] += 1
                    if self.args.krylov == 'lgmres':
                        delta, info = lgmres(Aop, -r0, rtol=self.args.gmres_tol,
                                             atol=0.0, maxiter=self.args.gmres_iters,
                                             callback=cb)
                    else:
                        Mop = None
                        if frozen_solve is not None and target > 0.0:
                            scale = -1.0 / (target * self.dt)
                            Mop = LinearOperator(
                                (self.n_unknown, self.n_unknown),
                                matvec=lambda vv: scale * frozen_solve(vv),
                                dtype=np.float64,
                            )
                        delta, info = gmres(Aop, -r0, M=Mop,
                                            rtol=self.args.gmres_tol,
                                            atol=0.0, restart=self.args.gmres_restart,
                                            maxiter=self.args.gmres_iters,
                                            callback=cb, callback_type='legacy')
                last_gmres_info = int(info)
                last_gmres_it = gmres_counter['n']

                # Pressure-increment trust region inherited from the user's solver.
                dmax = float(np.max(np.abs(delta))) if delta.size else 0.0
                pscale = max(1.0, float(np.max(np.abs(p_stage))) if p_stage.size else 0.0)
                dlimit = max(self.args.max_delta_p_abs, self.args.max_delta_p_frac * pscale)
                if dmax > dlimit and dmax > 0.0:
                    delta *= dlimit / dmax

                accepted = False
                best = rel0
                for beta in self.beta_grid:
                    if beta == 0.0:
                        continue
                    krylov_failed = (self.args.newton_linear_solver == 'gmres' and info != 0)
                    if krylov_failed and abs(beta) > self.args.max_beta_if_gmres_fail:
                        continue
                    cand = p_stage + beta * delta
                    rc = res_np(cand, target)
                    relc = float(np.linalg.norm(rc) / source_norm)
                    if not np.isfinite(relc):
                        continue
                    dvals = diag_jit(jnp.asarray(cand, dtype=self.dtype), float(target))
                    dispmax = float(np.asarray(dvals[9]))
                    if dispmax > self.args.max_pressure_displacement_frac * self.dx:
                        continue
                    if relc < best * (1.0 - self.args.min_improve):
                        p_stage = cand
                        best = relc
                        accepted = True
                        last_beta = beta
                        accepted_newton += 1
                        break
                if not accepted:
                    break

            if success:
                p_u = p_stage
                lam = target
                lam_step = min(self.args.lambda_step_max, lam_step * self.args.lambda_growth)
            else:
                lam_step *= 0.5
                reductions += 1
                if lam_step < self.args.lambda_step_min:
                    raise StepSolveFailure(
                        f'CCOP continuation failed at step={step}, lambda={lam:.6f}, '
                        f'target={target:.6f}, rel={last_rel:.3e}, gmres_info={last_gmres_info}'
                    )

        # Strict final full residual check.
        r_final = res_np(p_u, 1.0)
        rel_final = float(np.linalg.norm(r_final) / source_norm)
        if rel_final > self.args.final_accept_rel * (1.0 + self.args.accept_rel_slack):
            raise StepSolveFailure(
                f'CCOP final residual failed at step={step}: rel={rel_final:.3e} '
                f'> tol={self.args.final_accept_rel:.3e}'
            )

        d = [np.asarray(v) for v in diag_jit(jnp.asarray(p_u, dtype=self.dtype), 1.0)]
        xT, uT, p_full = map(lambda z: np.asarray(z, dtype=np.float64), d[12:15])
        g = np.asarray(d[15], dtype=np.float64)
        geom_res_rel = float(d[11])
        if geom_res_rel > self.args.geometry_accept_rel:
            raise StepSolveFailure(
                f'CCOP geometry closure failed at step={step}: '
                f'geom_res/dx={geom_res_rel:.3e} > {self.args.geometry_accept_rel:.3e}'
            )

        diag = {
            'max_D_star': float(d[0]), 'rms_D_star': float(d[1]),
            'max_D_repr': float(d[2]), 'rms_D_repr': float(d[3]),
            'max_D_terminal': float(d[4]), 'rms_D_terminal': float(d[5]),
            'rms_D_frozen_after': float(d[6]),
            'rel_rms': rel_final,
            'pmax': float(d[7]), 'gmax': float(d[8]), 'dispmax': float(d[9]),
            'p_center': float(d[10]), 'geometry_resid_rel': geom_res_rel,
            'linear_resid_rel': np.nan,
            'lam': 1.0, 'lambda_reductions': float(reductions),
            'accepted_newton': float(accepted_newton),
            'gmres_info': float(last_gmres_info), 'gmres_it': float(last_gmres_it),
            'beta': float(last_beta), 'wall_time': time.time() - t0,
            'rho_min': self.rho0, 'rho_max': self.rho0,
            'warmstart_rel': float(warm_rel) if np.isfinite(warm_rel) else np.nan,
            'warmstart_it': float(warm_it), 'warmstart_info': float(warm_info),
        }
        diag['p_center_fit'] = self.center_pressure_fit(xT, p_full)
        return StepResult(xT, uT, p_full, diag)

    # ------------------------------------------------------------------
    # Matched conventional ISPH: same D/G, frozen x*
    # ------------------------------------------------------------------

    def solve_matched_one_step(self, step: int, x_n_np: np.ndarray,
                               u_n_np: np.ndarray, p_prev_full: np.ndarray) -> StepResult:
        """Frozen-geometry projection using the *same* common D/G as CCOP.

        The collocated composition D_h G_h may possess near-null/checkerboard
        modes.  We therefore solve the linear system in least-squares form by
        with GMRES by default; an exact-transpose LSMR fallback is available.  This does not hide a
        residual floor: the achieved floor is reported as linear_resid_rel.
        Use --strict-linear-accept to turn a nonzero floor into a hard failure.
        """
        self._prepare_step_graph(x_n_np)
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = jnp.asarray(x_star, dtype=self.dtype)
        u_star = jnp.asarray(u_star, dtype=self.dtype)

        t0 = time.time()
        p_u, source_full, lin_rel, info, itn = self._solve_frozen_common_pressure(
            x_star, u_star, p_prev_full, self.args.linear_tol,
            self.args.linear_iters
        )

        if lin_rel > self.args.linear_accept_rel:
            msg = (
                f'matched-ISPH D/G residual floor step={step}: rel={lin_rel:.3e} '
                f'> requested={self.args.linear_accept_rel:.3e}; solver_info={info}'
            )
            if self.args.strict_linear_accept:
                raise StepSolveFailure(msg)
            print('[matched-DG-warning]', msg)

        p_full_j = self._unpack_jax(jnp.asarray(p_u, dtype=self.dtype))
        g = self._G_common(x_star, p_full_j)
        uT = u_star - self.dt * g
        xT = x_star - self.args.position_correction * self.dt * self.dt * g

        drepr = self._D_common(x_star, uT)
        dterminal = self._D_common(xT, uT)
        disp = jnp.sqrt(jnp.sum((xT-x_star)**2, axis=1))
        gnorm = jnp.sqrt(jnp.sum(g*g, axis=1))

        dstar_np = np.asarray(source_full, dtype=np.float64)[self.unknown_idx_np]
        drepr_np = np.asarray(drepr, dtype=np.float64)[self.unknown_idx_np]
        dterminal_np = np.asarray(dterminal, dtype=np.float64)[self.unknown_idx_np]
        xT_np = np.asarray(xT, dtype=np.float64)
        uT_np = np.asarray(uT, dtype=np.float64)
        p_full_np = np.asarray(p_full_j, dtype=np.float64)

        diag = {
            'max_D_star': float(np.max(np.abs(dstar_np))),
            'rms_D_star': float(np.sqrt(np.mean(dstar_np*dstar_np)+1e-30)),
            'max_D_repr': float(np.max(np.abs(drepr_np))),
            'rms_D_repr': float(np.sqrt(np.mean(drepr_np*drepr_np)+1e-30)),
            'max_D_terminal': float(np.max(np.abs(dterminal_np))),
            'rms_D_terminal': float(np.sqrt(np.mean(dterminal_np*dterminal_np)+1e-30)),
            'rms_D_frozen_after': float(np.sqrt(np.mean(drepr_np*drepr_np)+1e-30)),
            'rel_rms': lin_rel,
            'linear_resid_rel': lin_rel,
            'geometry_resid_rel': 0.0,
            'pmax': float(np.max(np.abs(p_full_np))),
            'gmax': float(np.max(np.asarray(gnorm))),
            'dispmax': float(np.max(np.asarray(disp))),
            'p_center': float(p_full_np[self.center_idx]),
            'lam': 1.0, 'lambda_reductions': 0.0, 'accepted_newton': 0.0,
            'gmres_info': float(info), 'gmres_it': float(itn),
            'beta': 1.0, 'wall_time': time.time()-t0,
            'rho_min': self.rho0, 'rho_max': self.rho0,
        }
        diag['p_center_fit'] = self.center_pressure_fit(xT_np, p_full_np)
        return StepResult(xT_np, uT_np, p_full_np, diag)

    # ------------------------------------------------------------------
    # SISPH-paper stencil branch
    # ------------------------------------------------------------------

    def solve_sisph_paper_one_step(self, step: int, x_n_np: np.ndarray,
                                   u_n_np: np.ndarray, p_prev_full: np.ndarray) -> StepResult:
        self._prepare_step_graph(x_n_np)
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = jnp.asarray(x_star, dtype=self.dtype)
        u_star = jnp.asarray(u_star, dtype=self.dtype)

        if self.args.paper_density == 'summation':
            rho = self._density_paper(x_star)
        else:
            rho = jnp.full((self.N,), self.rho0, dtype=self.dtype)
        rho_np = np.asarray(rho, dtype=np.float64)

        # Default controlled test: exact same p=0 mask as CCOP/matched-ISPH.
        paper_free = self.free_mask_np.copy()
        if self.args.paper_density_cutoff:
            paper_free |= (rho_np / self.rho0 < self.args.paper_rho_cutoff)
            if not np.any(paper_free):
                paper_free[self.center_idx] = True
        paper_unknown_np = np.where(~paper_free)[0].astype(np.int32)
        paper_unknown = jnp.asarray(paper_unknown_np)
        npu = len(paper_unknown_np)
        if npu == 0:
            raise StepSolveFailure('sisph-paper: no pressure unknowns after free-surface mask')

        @jax.jit
        def unpack_paper(p_u):
            p = jnp.zeros((self.N,), dtype=self.dtype)
            return p.at[paper_unknown].set(p_u, unique_indices=True)

        Dstar_paper = self._D_paper(x_star, u_star, rho)
        rhs = Dstar_paper[paper_unknown] / self.dt
        rhs_np = np.asarray(rhs, dtype=np.float64)
        rhs_norm = max(1e-30, float(np.linalg.norm(rhs_np)))

        @jax.jit
        def A_apply(p_u):
            p_full = unpack_paper(p_u)
            return self._A_paper_full(x_star, p_full, rho)[paper_unknown]

        def matvec(v):
            return np.array(A_apply(jnp.asarray(v, dtype=self.dtype)), dtype=np.float64, copy=True)

        Aop = LinearOperator((npu, npu), matvec=matvec, dtype=np.float64)
        x0 = p_prev_full[paper_unknown_np].astype(np.float64) \
            if self.args.pressure_init == 'previous' else None
        counter = {'n': 0}
        def cb(_): counter['n'] += 1
        t0 = time.time()
        p_u, info = gmres(Aop, rhs_np, x0=x0, rtol=self.args.linear_tol, atol=0.0,
                          restart=self.args.gmres_restart, maxiter=self.args.linear_iters,
                          callback=cb, callback_type='legacy')

        lin_r = matvec(p_u) - rhs_np
        lin_rel = float(np.linalg.norm(lin_r) / rhs_norm)
        if info != 0 or lin_rel > self.args.linear_accept_rel:
            raise StepSolveFailure(
                f'sisph-paper PPE solve failed step={step}: info={info}, rel={lin_rel:.3e}'
            )

        p_full = unpack_paper(jnp.asarray(p_u, dtype=self.dtype))
        g = self._G_paper(x_star, p_full, rho)
        uT = u_star - self.dt * g
        xT = x_star - self.args.position_correction * self.dt * self.dt * g

        # Branch-native represented divergence and common apples-to-apples terminal diagnostic.
        drepr = self._D_paper(x_star, uT, rho)
        dterminal_common = self._D_common(xT, uT)
        dstar_common = self._D_common(x_star, u_star)
        dfrozen_common = self._D_common(x_star, uT)
        disp = jnp.sqrt(jnp.sum((xT-x_star)**2, axis=1))
        gnorm = jnp.sqrt(jnp.sum(g*g, axis=1))

        xT_np = np.asarray(xT, dtype=np.float64)
        uT_np = np.asarray(uT, dtype=np.float64)
        p_full_np = np.asarray(p_full, dtype=np.float64)
        drepr_np = np.asarray(drepr, dtype=np.float64)[paper_unknown_np]
        dterminal_np = np.asarray(dterminal_common, dtype=np.float64)[self.unknown_idx_np]
        dstar_np = np.asarray(dstar_common, dtype=np.float64)[self.unknown_idx_np]
        dfrozen_np = np.asarray(dfrozen_common, dtype=np.float64)[self.unknown_idx_np]

        diag = {
            'max_D_star': float(np.max(np.abs(dstar_np))),
            'rms_D_star': float(np.sqrt(np.mean(dstar_np*dstar_np)+1e-30)),
            'max_D_repr': float(np.max(np.abs(drepr_np))),
            'rms_D_repr': float(np.sqrt(np.mean(drepr_np*drepr_np)+1e-30)),
            'max_D_terminal': float(np.max(np.abs(dterminal_np))),
            'rms_D_terminal': float(np.sqrt(np.mean(dterminal_np*dterminal_np)+1e-30)),
            'rms_D_frozen_after': float(np.sqrt(np.mean(dfrozen_np*dfrozen_np)+1e-30)),
            'rel_rms': lin_rel,
            'linear_resid_rel': lin_rel,
            'geometry_resid_rel': 0.0,
            'pmax': float(np.max(np.abs(p_full_np))),
            'gmax': float(np.max(np.asarray(gnorm))),
            'dispmax': float(np.max(np.asarray(disp))),
            'p_center': float(p_full_np[self.center_idx]),
            'lam': 1.0, 'lambda_reductions': 0.0, 'accepted_newton': 0.0,
            'gmres_info': float(info), 'gmres_it': float(counter['n']),
            'beta': 1.0, 'wall_time': time.time()-t0,
            'rho_min': float(rho_np.min()), 'rho_max': float(rho_np.max()),
        }
        diag['p_center_fit'] = self.center_pressure_fit(xT_np, p_full_np)
        return StepResult(xT_np, uT_np, p_full_np, diag)

    def solve_one_step(self, step: int, x_n_np: np.ndarray,
                       u_n_np: np.ndarray, p_prev_full: np.ndarray) -> StepResult:
        if self.method == 'ccop':
            result = self.solve_ccop_one_step(step, x_n_np, u_n_np, p_prev_full)
        elif self.method == 'matched-isph':
            result = self.solve_matched_one_step(step, x_n_np, u_n_np, p_prev_full)
        elif self.method == 'sisph-paper':
            result = self.solve_sisph_paper_one_step(step, x_n_np, u_n_np, p_prev_full)
        else:
            raise ValueError(self.method)
        return self._augment_step_diagnostics(result, x_n_np, u_n_np)

    # ------------------------------------------------------------------
    # Diagnostics / output
    # ------------------------------------------------------------------

    def center_pressure_fit(self, x: np.ndarray, p: np.ndarray,
                            radius_factor: float | None = None) -> float:
        rf = self.args.center_fit_radius if radius_factor is None else radius_factor
        if rf <= 0.0:
            return float(p[self.center_idx])
        r = np.linalg.norm(x, axis=1)
        idx = np.where(r <= rf * self.R)[0]
        if len(idx) < 6:
            return float(p[self.center_idx])
        xx = x[idx, 0]
        yy = x[idx, 1]
        A = np.column_stack([np.ones_like(xx), xx, yy, xx*xx, xx*yy, yy*yy])
        try:
            coef, *_ = np.linalg.lstsq(A, p[idx], rcond=None)
            return float(coef[0])
        except Exception:
            return float(p[self.center_idx])

    def compute_energy(self, x: np.ndarray, u: np.ndarray):
        K = 0.5 * self.rho0 * self.V0 * float(np.sum(u*u))
        V = 0.5 * self.rho0 * self.V0 * self.omega2 * float(np.sum(x*x))
        return K, V, K+V

    def run(self):
        x, u, p = self.initial_state()
        xs, us, ps, times = [], [], [], []
        center_p, center_pref, center_p_fit = [], [], []
        center_p_err_abs, center_p_err_rel = [], []
        center_p_fit_err_abs, center_p_fit_err_rel = [], []
        rels, repr_rms, terminal_rms, frozen_rms = [], [], [], []
        pmaxs, gmaxs, geom_resids, linear_resids = [], [], [], []
        rho_mins, rho_maxs = [], []
        K_saved, V_saved, E_saved, Erel_saved = [], [], [], []
        Eerr_abs_saved = []
        Dcs0_saved, Dcs1_saved, DcT_saved, DcTr_saved = [], [], [], []
        Dns0_saved, Dnr_saved, DnTr_saved = [], [], []
        DcT_all_saved, DcT_free_saved = [], []
        DnT_all_saved, DnT_free_saved = [], []
        missing_edges_saved = []
        pred_Erel_saved, pressure_Erel_saved, step_Erel_saved = [], [], []
        substeps_saved, dt_last_saved, t_p_saved = [], [], []

        dt_nominal = self.dt_nominal
        t_now = 0.0
        dt_sub_next = dt_nominal * self.args.dt_substep_init_factor
        dt_min = dt_nominal * self.args.dt_min_factor

        K, V, E = self.compute_energy(x, u)
        E0 = E
        print(f'step=00000, method={self.method}, t=0, K={K:.8e}, V={V:.8e}, E={E:.8e}')
        if self.args.audit_discrete_parity:
            self.audit_discrete_parity(x, u)
        xs.append(x.copy()); us.append(u.copy()); ps.append(p.copy()); times.append(0.0)
        center_p.append(p[self.center_idx])
        center_pref.append(reference_center_pressure(0.0, self.R, self.args.delta0,
                                                      self.args.omega, self.rho0))
        center_p_fit.append(self.center_pressure_fit(x, p))
        # No pressure solve has been performed at t=0, so a pressure error there
        # would be misleading.  Store NaN until the first accepted pressure solve.
        center_p_err_abs.append(np.nan); center_p_err_rel.append(np.nan)
        center_p_fit_err_abs.append(np.nan); center_p_fit_err_rel.append(np.nan)
        rels.append(np.nan); repr_rms.append(np.nan); terminal_rms.append(np.nan); frozen_rms.append(np.nan)
        pmaxs.append(np.nan); gmaxs.append(np.nan); geom_resids.append(np.nan); linear_resids.append(np.nan)
        rho_mins.append(np.nan); rho_maxs.append(np.nan)
        K_saved.append(K); V_saved.append(V); E_saved.append(E); Erel_saved.append(0.0)
        Eerr_abs_saved.append(0.0)
        Dcs0_saved.append(np.nan); Dcs1_saved.append(np.nan); DcT_saved.append(np.nan); DcTr_saved.append(np.nan)
        Dns0_saved.append(np.nan); Dnr_saved.append(np.nan); DnTr_saved.append(np.nan)
        DcT_all_saved.append(np.nan); DcT_free_saved.append(np.nan)
        DnT_all_saved.append(np.nan); DnT_free_saved.append(np.nan)
        missing_edges_saved.append(np.nan)
        pred_Erel_saved.append(np.nan); pressure_Erel_saved.append(np.nan); step_Erel_saved.append(np.nan)
        substeps_saved.append(0); dt_last_saved.append(np.nan); t_p_saved.append(0.0)

        def write_output_checkpoint(tag='latest'):
            np.savez(
                self.args.output,
                method=np.array([self.method]), kernel=np.array([self.kernel]),
                x=np.asarray(xs), u=np.asarray(us), p=np.asarray(ps), times=np.asarray(times),
                center_pressure=np.asarray(center_p), center_pressure_fit=np.asarray(center_p_fit),
                center_pressure_ref=np.asarray(center_pref),
                center_pressure_error_abs=np.asarray(center_p_err_abs),
                center_pressure_error_rel=np.asarray(center_p_err_rel),
                center_pressure_fit_error_abs=np.asarray(center_p_fit_err_abs),
                center_pressure_fit_error_rel=np.asarray(center_p_fit_err_rel),
                rel_rms=np.asarray(rels),
                rms_D_repr=np.asarray(repr_rms), rms_D_terminal=np.asarray(terminal_rms),
                rms_D_frozen_after=np.asarray(frozen_rms), pmax=np.asarray(pmaxs),
                gmax=np.asarray(gmaxs), geometry_resid_rel=np.asarray(geom_resids),
                linear_resid_rel=np.asarray(linear_resids), rho_min=np.asarray(rho_mins),
                rho_max=np.asarray(rho_maxs),
                kinetic_energy=np.asarray(K_saved), potential_energy=np.asarray(V_saved),
                total_energy=np.asarray(E_saved),
                total_energy_error_abs=np.asarray(Eerr_abs_saved),
                total_energy_error_rel=np.asarray(Erel_saved),
                # Backward-compatible alias retained for old plotting scripts.
                total_energy_rel=np.asarray(Erel_saved),
                rms_D_common_star_before=np.asarray(Dcs0_saved),
                rms_D_common_star_after=np.asarray(Dcs1_saved),
                rms_D_common_terminal_frozen=np.asarray(DcT_saved),
                rms_D_common_terminal_rebuilt=np.asarray(DcTr_saved),
                rms_D_native_star_before=np.asarray(Dns0_saved),
                rms_D_native_repr_after=np.asarray(Dnr_saved),
                rms_D_native_terminal_rebuilt=np.asarray(DnTr_saved),
                rms_D_common_terminal_all_rebuilt=np.asarray(DcT_all_saved),
                rms_D_common_terminal_free_rebuilt=np.asarray(DcT_free_saved),
                rms_D_native_terminal_all_rebuilt=np.asarray(DnT_all_saved),
                rms_D_native_terminal_free_rebuilt=np.asarray(DnT_free_saved),
                terminal_missing_directed_edges=np.asarray(missing_edges_saved),
                predictor_energy_rel=np.asarray(pred_Erel_saved),
                pressure_stage_energy_rel=np.asarray(pressure_Erel_saved),
                energy_step_rel=np.asarray(step_Erel_saved),
                substeps=np.asarray(substeps_saved),
                dt_last=np.asarray(dt_last_saved), pressure_times=np.asarray(t_p_saved),
                X=self.X_np, unknown_idx=self.unknown_idx_np, free_idx=self.free_idx_np,
                center_idx=np.array([self.center_idx]), dx=np.array([self.dx]), h=np.array([self.h]),
                rho0=np.array([self.rho0]), dt=np.array([dt_nominal]), radius=np.array([self.R]),
                initial_kinetic_energy=np.array([K_saved[0]]),
                initial_potential_energy=np.array([V_saved[0]]),
                initial_total_energy=np.array([E_saved[0]]),
                delta0=np.array([self.args.delta0]), omega=np.array([self.args.omega]),
                pressure_time_theta=np.array([self.effective_pressure_theta()]),
                paper_operator_mode=np.array([self.paper_operator_mode]),
                comparison_contract=np.array([
                    'shared mode: all branches use identical D/G, V0, rho0 and harness; sisph-paper differs only through Brookshaw PPE A_B instead of composed DG'
                ]),
                last_checkpoint_tag=np.array([str(tag)]),
            )

        time_eps = max(self.args.time_eps_abs, self.args.time_eps_factor * dt_nominal)

        for step in range(1, self.args.steps + 1):
            target_time = step * dt_nominal
            n_sub = 0
            last_result = None
            last_dt = np.nan
            last_t_p = t_now

            while True:
                remaining = target_time - t_now
                if remaining <= time_eps:
                    t_now = target_time
                    break
                if n_sub >= self.args.max_substeps_per_step:
                    write_output_checkpoint(tag=f'failure_step_{step:05d}')
                    raise RuntimeError(f'exceeded max_substeps_per_step at step={step}')

                dt_try = min(dt_sub_next, remaining)
                if dt_try <= 0.0:
                    raise RuntimeError('non-positive adaptive substep')
                if dt_try < dt_min and remaining > dt_min:
                    write_output_checkpoint(tag=f'failure_step_{step:05d}')
                    raise RuntimeError(
                        f'adaptive dt below dt_min at step={step}: {dt_try:.3e} < {dt_min:.3e}'
                    )

                old_dt = self.dt
                self.dt = float(dt_try)
                try:
                    result = self.solve_one_step(step, x, u, p)
                except StepSolveFailure as e:
                    self.dt = old_dt
                    if dt_try <= dt_min * (1.0 + 1e-10) and remaining > dt_min:
                        write_output_checkpoint(tag=f'failure_step_{step:05d}')
                        raise RuntimeError(
                            f'{self.method} failed at dt_min on step={step}: {e}'
                        ) from e
                    dt_sub_next = max(0.5 * dt_try, dt_min)
                    print(
                        f'[adaptive-substep] reject method={self.method}, step={step:05d}, '
                        f't={t_now:.8e}, dt={dt_try:.3e}; retry={dt_sub_next:.3e}; reason={e}'
                    )
                    continue
                finally:
                    self.dt = old_dt

                x, u, p = result.x, result.u, result.p_full
                last_result = result
                last_dt = dt_try
                last_t_p = t_now + self.effective_pressure_theta() * dt_try
                t_now += dt_try
                n_sub += 1
                dt_sub_next = min(dt_nominal, max(dt_min, dt_try * self.args.dt_substep_growth))

            if last_result is None:
                raise RuntimeError(f'no accepted substep for physical step={step}')

            K, V, E = self.compute_energy(x, u)
            pref = reference_center_pressure(last_t_p, self.R, self.args.delta0,
                                             self.args.omega, self.rho0)
            if step == 1 or step % self.args.save_every == 0:
                d = last_result.diag
                p_center = float(d['p_center'])
                p_center_fit = float(d['p_center_fit'])
                p_err_abs = p_center - pref
                p_err_rel = p_err_abs / pref if abs(pref) > 1e-30 else np.nan
                p_fit_err_abs = p_center_fit - pref
                p_fit_err_rel = p_fit_err_abs / pref if abs(pref) > 1e-30 else np.nan
                E_err_abs = E - E0
                E_err_rel = E_err_abs / E0 if abs(E0) > 1e-30 else np.nan
                print(
                    f"step={step:05d}, method={self.method}, t={t_now:.8e}, "
                    f"K={K:.8e}, V={V:.8e}, E={E:.8e}, dE={E_err_abs:+.3e}, dE/E0={E_err_rel:+.3e}, "
                    f"p_center/ref={p_center:.8e}/{pref:.8e}, "
                    f"p_err={p_err_abs:+.3e}, p_err_rel={p_err_rel:+.3e}, "
                    f"p_fit={p_center_fit:.8e}, p_fit_err_rel={p_fit_err_rel:+.3e}, "
                    f"Dnative*:={d['rms_D_native_star_before']:.3e}, "
                    f"Dnative_repr={d['rms_D_native_repr_after']:.3e}, "
                    f"Dnative_T(int/all/free)={d['rms_D_native_terminal_rebuilt']:.3e}/"
                    f"{d['rms_D_native_terminal_all_rebuilt']:.3e}/"
                    f"{d['rms_D_native_terminal_free_rebuilt']:.3e}, "
                    f"Dcommon_T(frozen/rebuild)={d['rms_D_common_terminal_frozen']:.3e}/"
                    f"{d['rms_D_common_terminal_rebuilt']:.3e}, missing_edges={int(d['terminal_missing_directed_edges'])}, "
                    f"solve_rel={d['rel_rms']:.3e}, geom/dx={d['geometry_resid_rel']:.3e}, "
                    f"pmax={d['pmax']:.3e}, "
                    f"gmres_it={int(d['gmres_it'])}, info={int(d['gmres_info'])}, substeps={n_sub}, wall={d['wall_time']:.2f}s"
                )
                xs.append(x.copy()); us.append(u.copy()); ps.append(p.copy()); times.append(t_now)
                center_p.append(p_center); center_p_fit.append(p_center_fit); center_pref.append(pref)
                center_p_err_abs.append(p_err_abs); center_p_err_rel.append(p_err_rel)
                center_p_fit_err_abs.append(p_fit_err_abs); center_p_fit_err_rel.append(p_fit_err_rel)
                rels.append(d['rel_rms']); repr_rms.append(d['rms_D_repr']); terminal_rms.append(d['rms_D_terminal'])
                frozen_rms.append(d['rms_D_frozen_after']); pmaxs.append(d['pmax']); gmaxs.append(d['gmax'])
                geom_resids.append(d['geometry_resid_rel']); linear_resids.append(d['linear_resid_rel'])
                rho_mins.append(d['rho_star_min']); rho_maxs.append(d['rho_star_max'])
                K_saved.append(K); V_saved.append(V); E_saved.append(E); Erel_saved.append(E_err_rel)
                Eerr_abs_saved.append(E_err_abs)
                Dcs0_saved.append(d['rms_D_common_star_before'])
                Dcs1_saved.append(d['rms_D_common_star_after'])
                DcT_saved.append(d['rms_D_common_terminal_frozen'])
                DcTr_saved.append(d['rms_D_common_terminal_rebuilt'])
                Dns0_saved.append(d['rms_D_native_star_before'])
                Dnr_saved.append(d['rms_D_native_repr_after'])
                DnTr_saved.append(d['rms_D_native_terminal_rebuilt'])
                DcT_all_saved.append(d['rms_D_common_terminal_all_rebuilt'])
                DcT_free_saved.append(d['rms_D_common_terminal_free_rebuilt'])
                DnT_all_saved.append(d['rms_D_native_terminal_all_rebuilt'])
                DnT_free_saved.append(d['rms_D_native_terminal_free_rebuilt'])
                missing_edges_saved.append(d['terminal_missing_directed_edges'])
                pred_Erel_saved.append(d['predictor_energy_rel'])
                pressure_Erel_saved.append(d['pressure_stage_energy_rel'])
                step_Erel_saved.append(d['energy_step_rel'])
                substeps_saved.append(n_sub); dt_last_saved.append(last_dt); t_p_saved.append(last_t_p)

                if step == 1 or step % self.args.checkpoint_every == 0:
                    write_output_checkpoint(tag=f'step_{step:05d}')
                    print('checkpoint saved:', self.args.output)

        write_output_checkpoint(tag='final')
        print('saved:', self.args.output)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Controlled CCOP / matched-ISPH / SISPH-paper comparison harness.'
    )
    ap.add_argument('--method', choices=['ccop', 'matched-isph', 'sisph-paper'], default='ccop')
    ap.add_argument('--backend', choices=['auto', 'cpu', 'gpu'], default='auto')
    ap.add_argument('--fp64', action='store_true')

    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--resolution', type=int, default=100)
    ap.add_argument('--init-scheme', choices=['cartesian', 'ring'], default='ring')
    ap.add_argument('--phase-jitter', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--add-bbox-points', action='store_true')
    ap.add_argument('--n-theta-outer', type=int, default=0)
    ap.add_argument('--dx', type=float, default=-1.0)

    ap.add_argument('--kernel', choices=['cubic', 'quintic'], default='quintic',
                    help='Common kernel. quintic matches the default PySPH SISPH kernel family.')
    ap.add_argument('--h-factor', type=float, default=1.2)
    ap.add_argument('--graph-skin', type=float, default=0.25,
                    help='Extra frozen-neighbor skin in units of h; kernel remains compactly supported.')
    ap.add_argument('--common-operator', choices=['raw', 'corrected'], default='corrected',
                    help='Shared CCOP/matched D/G. corrected enforces first-order moment completeness at each configuration.')
    ap.add_argument('--common-correction-reg', type=float, default=1e-10)
    ap.add_argument('--min-neighbors-warn', type=int, default=8)

    ap.add_argument('--rho0', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=2.5e-3)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--checkpoint-every', type=int, default=1000)

    ap.add_argument('--delta0', type=float, default=0.4)
    ap.add_argument('--omega', type=float, default=1.2)
    ap.add_argument('--omega2', type=float, default=-1.0)
    ap.add_argument('--predictor', choices=['kinematic', 'semi-implicit', 'implicit-midpoint-force'],
                    default='implicit-midpoint-force')
    ap.add_argument('--position-correction', type=float, default=0.5)

    ap.add_argument('--free-surface-mode', choices=['radial', 'material', 'none'], default='material')
    ap.add_argument('--free-surface-width', type=float, default=1.0)
    ap.add_argument('--gauge-index', type=int, default=-1)

    # CCOP geometry/action placement.
    ap.add_argument('--placement', choices=['midpoint', 'star', 'current', 'terminal'], default='midpoint')
    ap.add_argument('--theta', type=float, default=0.5)
    ap.add_argument('--geometry-iters', type=int, default=20)
    ap.add_argument('--geometry-tol-rel', type=float, default=1e-11,
                    help='Early-freeze criterion for fixed-point geometry iterations, normalized by dx.')
    ap.add_argument('--geometry-accept-rel', type=float, default=1e-8,
                    help='Reject CCOP step if final geometry fixed-point residual/dx exceeds this.')

    # CCOP nonlinear solve.
    ap.add_argument('--ccop-warmstart', choices=['matched', 'previous', 'zero'], default='matched',
                    help='Solver-only initialization; matched solves frozen common D/G before nonlinear CCOP correction.')
    ap.add_argument('--warmstart-tol', type=float, default=1e-8)
    ap.add_argument('--warmstart-iters', type=int, default=1200)
    ap.add_argument('--lambda-step', type=float, default=1.0)
    ap.add_argument('--lambda-step-max', type=float, default=1.0)
    ap.add_argument('--lambda-step-min', type=float, default=1e-4)
    ap.add_argument('--lambda-complete-tol', type=float, default=1e-10)
    ap.add_argument('--lambda-growth', type=float, default=1.4)
    ap.add_argument('--lambda-accept-rel', type=float, default=1e-8)
    ap.add_argument('--final-accept-rel', type=float, default=1e-8)
    ap.add_argument('--accept-rel-slack', type=float, default=0.05)
    ap.add_argument('--newton-iters', type=int, default=10)
    ap.add_argument('--krylov', choices=['gmres', 'lgmres'], default='gmres')
    ap.add_argument('--gmres-iters', type=int, default=300)
    ap.add_argument('--gmres-restart', type=int, default=100)
    ap.add_argument('--gmres-tol', type=float, default=1e-10)
    ap.add_argument('--jvp-mode', choices=['autodiff', 'fd'], default='autodiff')
    ap.add_argument('--jvp-eps', type=float, default=1e-6)
    ap.add_argument('--newton-damp', type=float, default=0.0)
    ap.add_argument('--newton-linear-solver', choices=['lsmr', 'gmres'], default='gmres',
                    help='GMRES is faster for the corrected common operator; LSMR is a rank-deficient fallback.')
    ap.add_argument('--newton-lsmr-damp', type=float, default=0.0)
    ap.add_argument('--max-delta-p-frac', type=float, default=0.5)
    ap.add_argument('--max-delta-p-abs', type=float, default=0.1)
    ap.add_argument('--max-beta-if-gmres-fail', type=float, default=0.25)
    ap.add_argument('--beta-grid', type=str, default='1,0.75,0.5,0.25,0.125,0.0625,0')
    ap.add_argument('--min-improve', type=float, default=1e-5)
    ap.add_argument('--max-pressure-displacement-frac', type=float, default=0.2)

    # Linear branches.
    ap.add_argument('--linear-tol', type=float, default=1e-10)
    ap.add_argument('--linear-accept-rel', type=float, default=1e-8)
    ap.add_argument('--linear-iters', type=int, default=300)
    ap.add_argument('--matched-linear-solver', choices=['sparse-lu', 'dense', 'lsmr', 'gmres'], default='sparse-lu')
    ap.add_argument('--matched-tikhonov', type=float, default=0.0,
                    help='Optional Tikhonov damping for matched D/G least-squares solve.')
    ap.add_argument('--matched-conlim', type=float, default=1e12)
    ap.add_argument('--ccop-preconditioner', choices=['none', 'frozen-dg'], default='frozen-dg',
                    help='Use a dense LU of frozen D(x*)G(x*) only as a Newton-GMRES preconditioner.')
    ap.add_argument('--preconditioner-shift', type=float, default=1e-12,
                    help='Relative diagonal shift for dense frozen-DG solves/factorizations.')
    ap.add_argument('--strict-linear-accept', action='store_true',
                    help='Hard-fail if a linear branch cannot reach --linear-accept-rel.')

    # SISPH-paper stencil options.
    ap.add_argument('--paper-operator-mode', choices=['shared', 'original'], default='shared',
                    help='shared: identical common D/G and fixed rho0/V0; only Brookshaw PPE differs. original: raw paper D/G and optional summation density.')
    ap.add_argument('--paper-density', choices=['summation', 'rho0'], default='rho0')
    ap.add_argument('--paper-density-floor', type=float, default=0.05)
    ap.add_argument('--paper-eps', type=float, default=1e-12)
    ap.add_argument('--paper-density-cutoff', action='store_true',
                    help='Additionally apply the SISPH-style rho/rho0 free-surface cutoff.')
    ap.add_argument('--paper-rho-cutoff', type=float, default=0.8)
    ap.add_argument('--audit-discrete-parity', action='store_true',
                    help='At t=0 print D/G identity errors and the Brookshaw-A versus composed-DG mismatch.')

    ap.add_argument('--pressure-init', choices=['previous', 'zero'], default='previous')
    ap.add_argument('--pressure-time-theta', type=float, default=-1.0)
    ap.add_argument('--center-fit-radius', type=float, default=0.20)

    # Physical adaptive substepping, shared by all methods.
    ap.add_argument('--dt-substep-init-factor', type=float, default=1.0)
    ap.add_argument('--dt-substep-growth', type=float, default=1.25)
    ap.add_argument('--dt-min-factor', type=float, default=0.0625)
    ap.add_argument('--max-substeps-per-step', type=int, default=64)
    ap.add_argument('--time-eps-factor', type=float, default=1e-10)
    ap.add_argument('--time-eps-abs', type=float, default=1e-14)

    ap.add_argument('--output', type=str, default='')
    args = ap.parse_args()
    if not args.output:
        args.output = f'comparison_{args.method.replace("-", "_")}.npz'

    solver = ThreeBranchISPH(args)
    solver.run()


if __name__ == '__main__':
    main()
