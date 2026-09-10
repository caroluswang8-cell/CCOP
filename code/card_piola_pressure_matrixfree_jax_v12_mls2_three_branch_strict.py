"""
Strict MLS2 three-branch configuration-assignment comparison.

The original MLS2 solver is preserved separately.  This comparison driver keeps
its particles, quadratic material MLS derivative, pressure boundary space,
external-force predictor, and pressure-corrected position update, while exposing
exactly three configuration assignments:

    ccop-midpoint       D_T^q / G_{1/2}  (reference CCOP branch),
    star-star           D_star^q / G_star (matched ISPH-like control),
    terminal-terminal   D_T^q / G_T       (endpoint-action control).

The star-star branch is a genuinely frozen predictor-configuration pressure
solve.  It must not be confused with the legacy ``--placement star`` option,
which retained D_T and changed only the pressure-action configuration.

Default production case:

    radius = 0.5, resolution = 100  => R/dx = 50,
    dt = 0.002, steps = 15000       => final physical time = 30.

Strict additions relative to the legacy file include mandatory fp64, exact JAX
JVP for nonlinear branches, a converged-geometry hard audit, raw accepted-state
Jacobian checks, a no-shift sparse-LU star-star solve, and separate anchor and
terminal residual histories.

This script solves the pressure-only nonlinear terminal constraint

    R(p) = (1 / J_T(p)) Div_X[ J_T(p) F_T(p)^(-1)
             (u* - lambda dt/rho0 F_theta(p)^(-T) Grad_X p) ] = 0

with a matrix-free Newton-Krylov method.  The production Jacobian-vector
product is computed from the complete nonlinear residual by exact JAX JVP;
centered/forward finite differences remain available only as diagnostics.

Therefore pressure-induced configuration changes, and the induced changes of
F_T, F_theta, J_T, G_theta and D_T, enter the Krylov direction.

This v6 variant treats the solved pressure as p^{n+theta}, not p^{n+1},
and adds strict adaptive physical substepping: if continuation cannot reach
lambda=1 for a proposed physical substep, the substep is rejected, the state
is rolled back, and the substep size is reduced.  Incomplete continuation is
never accepted unless --allow-incomplete-step is explicitly set for diagnostics.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, splu
from scipy.integrate import solve_ivp

import jax
import jax.numpy as jnp


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
    # stable order: center first-ish does not matter, but sorted helps reproducibility
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]




def generate_ring_disk(radius: float, dx: float, phase_jitter: bool = False, seed: int = 0,
                       add_bbox_points: bool = True, n_theta_outer: int = 0,
                       keep_center: bool = True, tol: float = 1e-10):
    """Concentric-ring material disk initialization with a strict outer ring.

    Returns
    -------
    X : (N,2) array
        Particle coordinates.
    outer_idx : int array
        Indices of the strict circular boundary ring plus optional bbox cardinal points.
    ring_id : int array
        0 for center/interior rings, 1e9 for strict outer ring, -1 for bbox points.

    This follows the user's damBreak_twoSets idea: interior rings stop before radius,
    and the free surface is a separately generated ring projected exactly to r=radius.
    """
    rng = np.random.default_rng(seed)
    # Match the user's rule: dr estimated from target point count.  Here dx is the
    # intended spatial spacing, so infer num_points by area/dx^2 and obtain dr~dx.
    dr = float(dx)
    dr = max(dr, 1e-12)

    xs_in, ys_in, ring_id_in = [], [], []
    if keep_center:
        xs_in.append(0.0); ys_in.append(0.0); ring_id_in.append(0)

    n_rings = int(np.floor(radius / dr))
    for k in range(1, n_rings + 1):
        r = k * dr
        if r >= radius:
            break
        n_theta = max(int(np.round(2.0 * np.pi * r / dr)), 6)
        if n_theta % 2 != 0:
            n_theta += 1
        offset = (rng.random() * 2.0 * np.pi) if phase_jitter else (0.5 * (k % 2) * 2.0*np.pi/n_theta)
        theta = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False) + offset
        xs_in.extend(r * np.cos(theta))
        ys_in.extend(r * np.sin(theta))
        ring_id_in.extend([k] * n_theta)

    xy_in = np.column_stack([xs_in, ys_in]).astype(np.float64) if xs_in else np.zeros((0,2), dtype=np.float64)
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

    # Use no random phase by default so the boundary has reproducible symmetry.
    offset = (rng.random() * 2.0*np.pi) if phase_jitter else 0.0
    theta = np.linspace(0.0, 2.0*np.pi, n_theta_outer, endpoint=False) + offset
    xy_out = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)]).astype(np.float64)
    rr = np.linalg.norm(xy_out, axis=1)
    xy_out *= (radius / rr)[:, None]
    ring_id_out = np.full(n_theta_outer, 10**9, dtype=np.int64)

    extras = []
    if add_bbox_points:
        extras = [[-radius, 0.0], [radius, 0.0], [0.0, -radius], [0.0, radius]]
    xy_extra = np.asarray(extras, dtype=np.float64).reshape((-1,2)) if extras else np.zeros((0,2), dtype=np.float64)
    ring_id_extra = -np.ones(len(xy_extra), dtype=np.int64)

    start_out = len(xy_in)
    X = np.vstack([xy_in, xy_out, xy_extra]).astype(np.float64)
    ring_id = np.concatenate([ring_id_in, ring_id_out, ring_id_extra])
    outer_idx = np.arange(start_out, start_out + len(xy_out), dtype=np.int32)
    if len(xy_extra) > 0:
        # Treat the four cardinal bbox points as material boundary/free pressure too.
        outer_idx = np.concatenate([outer_idx, np.arange(start_out + len(xy_out), len(X), dtype=np.int32)])

    rr_out = np.linalg.norm(X[outer_idx], axis=1)
    print(f'[ring-init] Generated points: {X.shape}, dr={dr:.8e}, boundary/free={len(outer_idx)}, '
          f'outer min r={rr_out.min():.16e}, outer max r={rr_out.max():.16e}, bbox={bool(add_bbox_points)}')
    return X, outer_idx.astype(np.int32), ring_id


def cubic_grad_2d(rvec: np.ndarray, h: float) -> np.ndarray:
    r = np.linalg.norm(rvec)
    if r < 1e-14 or r >= 2.0 * h:
        return np.zeros(2, dtype=np.float64)
    q = r / h
    sigma = 10.0 / (7.0 * math.pi * h * h)
    if q < 1.0:
        dWdr = sigma / h * (-3.0 * q + 2.25 * q * q)
    else:
        dWdr = sigma / h * (-0.75 * (2.0 - q) ** 2)
    return dWdr * (rvec / r)


def cubic_w_2d(r: float, h: float) -> float:
    q = r / h
    if q >= 2.0:
        return 0.0
    sigma = 10.0 / (7.0 * math.pi * h * h)
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q ** 3)
    return sigma * 0.25 * (2.0 - q) ** 3


def build_material_neighbors(X: np.ndarray, h: float, mls_reg: float) -> Dict[str, np.ndarray]:
    N = X.shape[0]
    tree = cKDTree(X)
    lists = tree.query_ball_point(X, 2.0 * h)
    # remove self
    neigh_lists = []
    maxn = 0
    for i, li in enumerate(lists):
        cur = [j for j in li if j != i]
        cur.sort()
        neigh_lists.append(cur)
        maxn = max(maxn, len(cur))

    neigh = np.zeros((N, maxn), dtype=np.int32)
    mask = np.zeros((N, maxn), dtype=np.float64)
    dX = np.zeros((N, maxn, 2), dtype=np.float64)
    gradW = np.zeros((N, maxn, 2), dtype=np.float64)
    wmls = np.zeros((N, maxn), dtype=np.float64)

    for i, cur in enumerate(neigh_lists):
        for k, j in enumerate(cur):
            rv = X[j] - X[i]
            neigh[i, k] = j
            mask[i, k] = 1.0
            dX[i, k] = rv
            gradW[i, k] = -cubic_grad_2d(rv, h)  # grad with respect to particle i
            wmls[i, k] = cubic_w_2d(np.linalg.norm(rv), h)
        # pad unused slots with self so gather is safe
        for k in range(len(cur), maxn):
            neigh[i, k] = i

    Minv = np.zeros((N, 2, 2), dtype=np.float64)
    eye = np.eye(2)
    for i in range(N):
        M = np.zeros((2, 2), dtype=np.float64)
        for k in range(maxn):
            if mask[i, k] == 0:
                continue
            dx = dX[i, k]
            M += wmls[i, k] * np.outer(dx, dx)
        M += mls_reg * (h ** 2) * eye
        Minv[i] = np.linalg.inv(M)
    return {
        'neigh': neigh,
        'mask': mask,
        'dX': dX,
        'gradW': gradW,
        'wmls': wmls,
        'Minv': Minv,
        'maxn': np.array([maxn], dtype=np.int32),
    }




def _mls_poly_basis(dX: np.ndarray, order: int) -> np.ndarray:
    """Polynomial basis for local MLS derivative reconstruction."""
    x = dX[:, 0]
    y = dX[:, 1]
    if order == 1:
        return np.column_stack([np.ones_like(x), x, y])
    if order == 2:
        return np.column_stack([np.ones_like(x), x, y, 0.5 * x * x, x * y, 0.5 * y * y])
    raise ValueError('MLS derivative order must be 1 or 2')


def build_mls_derivative_operator(X: np.ndarray, h: float, order: int, reg: float) -> Dict[str, np.ndarray]:
    """Build first/second-order MLS derivative weights on the material graph.

    The returned weights approximate material derivatives directly as

        df/dX_i = sum_j wx_ij f_j,
        df/dY_i = sum_j wy_ij f_j.

    For order=2 the weights reproduce quadratic scalar fields up to the local
    least-squares conditioning.  This is intended for the pressure operator,
    whose analytic oscillating-drop pressure is quadratic in space.

    Neighbors use the same compact support as the SPH material graph: r < 2h.
    The center particle itself is included in each local MLS fit, which improves
    constant reproduction and keeps the formula identical for boundary/center
    stencils.
    """
    N = X.shape[0]
    tree = cKDTree(X)
    lists = tree.query_ball_point(X, 2.0 * h)
    nbasis = 3 if order == 1 else 6
    min_nbr = nbasis + 4

    neigh_lists = []
    maxn = 0
    for i, li in enumerate(lists):
        cur = list(li)
        if i not in cur:
            cur.append(i)
        # Guarantee enough points by nearest-neighbor fallback.
        if len(cur) < min_nbr:
            rr = np.linalg.norm(X - X[i], axis=1)
            cur = list(np.argsort(rr)[:min_nbr])
        cur = sorted(set(int(j) for j in cur))
        neigh_lists.append(cur)
        maxn = max(maxn, len(cur))

    neigh = np.zeros((N, maxn), dtype=np.int32)
    mask = np.zeros((N, maxn), dtype=np.float64)
    dX = np.zeros((N, maxn, 2), dtype=np.float64)
    wx = np.zeros((N, maxn), dtype=np.float64)
    wy = np.zeros((N, maxn), dtype=np.float64)
    cond = np.zeros(N, dtype=np.float64)
    rank = np.zeros(N, dtype=np.int32)

    dx_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64) if order == 1 else np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dy_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64) if order == 1 else np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for i, cur in enumerate(neigh_lists):
        cur = np.asarray(cur, dtype=np.int32)
        m = len(cur)
        neigh[i, :m] = cur
        mask[i, :m] = 1.0
        dXi = X[cur] - X[i]
        dX[i, :m] = dXi
        rr = np.linalg.norm(dXi, axis=1)
        # Use the same cubic kernel weights as the existing material graph.
        w = np.asarray([cubic_w_2d(float(r), h) for r in rr], dtype=np.float64)
        # Ensure the center contributes even if the kernel normalization changes.
        w[rr < 1e-14] = np.maximum(w[rr < 1e-14], cubic_w_2d(0.0, h))
        B = _mls_poly_basis(dXi, order)
        A = (B.T * w) @ B
        tr = float(np.trace(A))
        lam = reg * (tr / max(nbasis, 1) + 1.0)
        Areg = A + lam * np.eye(nbasis)
        try:
            invA = np.linalg.inv(Areg)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f'Quadratic MLS moment matrix is singular at particle {i}; '
                'the strict comparison does not permit a pseudo-inverse fallback.'
            )
        M = invA @ (B.T * w)
        wx[i, :m] = dx_vec @ M
        wy[i, :m] = dy_vec @ M
        cond[i] = np.linalg.cond(Areg)
        rank[i] = np.linalg.matrix_rank(A, tol=1e-10)
        if m < maxn:
            neigh[i, m:] = i

    return {
        'neigh': neigh,
        'mask': mask,
        'dX': dX,
        'wx': wx,
        'wy': wy,
        'cond': cond,
        'rank': rank,
        'order': np.array([order], dtype=np.int32),
        'maxn': np.array([maxn], dtype=np.int32),
    }

def reference_center_pressure(t: float, R: float, delta0: float, Omega: float, rho0: float) -> float:
    # Integrate the affine ODE to exactly this scalar time.
    if t == 0.0:
        return 0.5 * rho0 * R * R * (delta0 * delta0 + Omega * Omega)

    def rhs(_t, y):
        a, delta = y
        rr = R ** 4 / a ** 4
        dd = ((rr - 1.0) / (rr + 1.0)) * (delta * delta + Omega * Omega)
        return [delta * a, dd]

    sol = solve_ivp(rhs, (0.0, t), [R, delta0], rtol=1e-10, atol=1e-12, method='DOP853')
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


class PiolaPressureSolver:
    def __init__(self, args):
        self.args = args
        if not bool(args.fp64):
            raise RuntimeError('This strict comparison requires --fp64 (float64 is mandatory).')
        jax.config.update('jax_enable_x64', bool(args.fp64))
        if args.backend != 'auto':
            jax.config.update('jax_platform_name', args.backend)
        self.dtype = jnp.float64 if args.fp64 else jnp.float32
        if not bool(jax.config.read('jax_enable_x64')):
            raise RuntimeError('JAX did not enable float64; refusing to run the strict comparison.')
        self.branch = str(args.branch)

        self.R = args.radius
        self.dx = args.dx if args.dx > 0 else (2.0 * args.radius / args.resolution)
        self.h = args.h_factor * self.dx
        self.V0 = self.dx * self.dx
        self.rho0 = args.rho0
        self.dt_nominal = args.dt
        self.dt = args.dt  # current substep size; may be temporarily changed by adaptive substepping
        self.omega2 = args.omega ** 2 if args.omega2 < 0 else args.omega2

        if args.init_scheme == 'ring':
            X, material_free_idx, ring_id = generate_ring_disk(
                args.radius, self.dx,
                phase_jitter=bool(args.phase_jitter),
                seed=int(args.seed),
                add_bbox_points=bool(args.add_bbox_points),
                n_theta_outer=int(args.n_theta_outer),
                keep_center=True,
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
            raise ValueError('free_surface_mode must be radial/material/none')
        self.free_mask_np = free
        self.unknown_idx_np = np.where(~free)[0].astype(np.int32)
        self.free_idx_np = np.where(free)[0].astype(np.int32)
        self.n_unknown = len(self.unknown_idx_np)

        neigh = build_material_neighbors(X, self.h, args.mls_reg)
        self.neigh_np = neigh
        self.use_mls_derivative_operator = args.reference_operator in ('mls1', 'mls2')

        if self.use_mls_derivative_operator:
            mls_order = 2 if args.reference_operator == 'mls2' else 1
            deriv = build_mls_derivative_operator(X, self.h, mls_order, args.mls2_reg)
            self.deriv_np = deriv
            print(f'[reference-operator] {args.reference_operator}: polynomial MLS derivative weights, '
                  f'order={mls_order}, maxn={int(deriv["maxn"][0])}, '
                  f'cond median={np.median(deriv["cond"]):.3e}, cond max={np.max(deriv["cond"]):.3e}, '
                  f'rank min={np.min(deriv["rank"])}')
        else:
            self.deriv_np = None

        # Optional first-order correction of the reference SPH gradient/divergence.
        # Raw SPH difference gradients on concentric rings are not exactly linearly
        # complete, which creates pressure ringing near the center and boundary.
        # The correction enforces, for each particle i,
        #     V0 * sum_j (X_j-X_i) \otimes gradW_corr_ij = I,
        # so gradients of linear scalar fields and divergences of linear vector fields
        # are reproduced exactly on the reference material graph.
        gradW_np = neigh['gradW'].copy()
        if args.reference_operator == 'corrected':
            dX_np = neigh['dX']
            mask_np = neigh['mask']
            A = self.V0 * np.einsum('nk,nka,nkb->nab', mask_np, dX_np, gradW_np)
            # Scale-aware Tikhonov regularization.  A should be close to identity;
            # the regularization only protects singular boundary/center stencils.
            eye = np.eye(2)[None, :, :]
            Areg = A + args.operator_correction_reg * eye
            invAT = np.transpose(np.linalg.inv(Areg), (0, 2, 1))
            gradW_np = np.einsum('ncb,nkb->nkc', invAT, gradW_np)
            # Useful diagnostic: corrected moment should be close to I.
            Ac = self.V0 * np.einsum('nk,nka,nkb->nab', mask_np, dX_np, gradW_np)
            err = np.linalg.norm(Ac - eye, axis=(1,2))
            print(f'[reference-operator] corrected: moment error max={err.max():.3e}, mean={err.mean():.3e}')
        elif args.reference_operator == 'raw':
            print('[reference-operator] raw SPH reference gradient/divergence')
        else:
            print(f'[reference-operator] {args.reference_operator}: using polynomial MLS derivative operator for D/G/F')

        # JAX constants
        self.X = jnp.asarray(X, dtype=self.dtype)
        self.neigh = jnp.asarray(neigh['neigh'])
        self.mask = jnp.asarray(neigh['mask'], dtype=self.dtype)
        self.dX = jnp.asarray(neigh['dX'], dtype=self.dtype)
        self.gradW = jnp.asarray(gradW_np, dtype=self.dtype)
        self.wmls = jnp.asarray(neigh['wmls'], dtype=self.dtype)
        self.Minv = jnp.asarray(neigh['Minv'], dtype=self.dtype)
        if self.use_mls_derivative_operator:
            self.deriv_neigh = jnp.asarray(self.deriv_np['neigh'])
            self.deriv_mask = jnp.asarray(self.deriv_np['mask'], dtype=self.dtype)
            self.deriv_wx = jnp.asarray(self.deriv_np['wx'], dtype=self.dtype)
            self.deriv_wy = jnp.asarray(self.deriv_np['wy'], dtype=self.dtype)
        else:
            self.deriv_neigh = self.neigh
            self.deriv_mask = self.mask
            self.deriv_wx = jnp.zeros_like(self.mask, dtype=self.dtype)
            self.deriv_wy = jnp.zeros_like(self.mask, dtype=self.dtype)
        self.unknown_idx = jnp.asarray(self.unknown_idx_np)
        self.free_mask = jnp.asarray(free)

        if not self.use_mls_derivative_operator or args.reference_operator != 'mls2':
            raise RuntimeError('The comparison is locked to the quadratic MLS2 derivative operator.')
        if int(np.min(self.deriv_np['rank'])) < 6:
            raise RuntimeError(
                f'Quadratic MLS moment rank loss: min rank={int(np.min(self.deriv_np["rank"]))} < 6.'
            )
        self.Dx_sp, self.Dy_sp = self._build_material_derivative_sparse()

        self.beta_grid = parse_float_list(args.beta_grid)
        if not self.beta_grid:
            self.beta_grid = [1.0, 0.5, 0.25, 0.125, 0.0]

        print('=== Strict MLS2 CCOP configuration-assignment comparison ===')
        print('JAX devices:', jax.devices())
        print('default backend:', jax.default_backend())
        print(f'N={self.N}, unknown={self.n_unknown}, free={len(self.free_idx_np)}')
        print(f'R={self.R}, dx={self.dx:.8e}, h={self.h:.8e}, dt={self.dt:.8e}, rho0={self.rho0}')
        print(f'branch={self.branch}, placement={args.placement}, theta={args.theta}, '
              f'pressure_time_theta={self.effective_pressure_theta()}, free_surface_mode={args.free_surface_mode}')
        print(f'R/dx={self.R/self.dx:.8g}, nominal final time={args.steps*self.dt_nominal:.8g}')


    def effective_pressure_theta(self) -> float:
        """Time level of the pressure variable p^{n+theta} for diagnostics.

        This is diagnostic only.  The pressure action geometry is still controlled
        by --placement/--theta.  For midpoint placement the pressure should be
        compared with the analytic pressure at t_n + 0.5 dt, not t_{n+1}.
        """
        if self.args.pressure_time_theta >= 0.0:
            return float(self.args.pressure_time_theta)
        if self.args.placement == 'midpoint':
            return 0.5
        if self.args.placement == 'current':
            return 0.0
        if self.args.placement == 'terminal':
            return 1.0
        if self.args.placement == 'star':
            # x* is a predictor, but for pressure-time diagnostics this is closer
            # to a terminal/predictor action than to current time.
            return 1.0
        return float(self.args.theta)

    def _build_material_derivative_sparse(self):
        """Sparse matrices for the exact fixed quadratic-MLS material derivative."""
        neigh = np.asarray(self.deriv_np['neigh'], dtype=np.int64)
        mask = np.asarray(self.deriv_np['mask'], dtype=np.float64)
        wx = np.asarray(self.deriv_np['wx'], dtype=np.float64) * mask
        wy = np.asarray(self.deriv_np['wy'], dtype=np.float64) * mask
        rows = np.repeat(np.arange(self.N, dtype=np.int64), neigh.shape[1])
        cols = neigh.reshape(-1)
        active = mask.reshape(-1) != 0.0
        Dx = coo_matrix(
            (wx.reshape(-1)[active], (rows[active], cols[active])),
            shape=(self.N, self.N), dtype=np.float64,
        ).tocsr()
        Dy = coo_matrix(
            (wy.reshape(-1)[active], (rows[active], cols[active])),
            shape=(self.N, self.N), dtype=np.float64,
        ).tocsr()
        one = np.ones(self.N, dtype=np.float64)
        const_defect = max(
            float(np.max(np.abs(Dx @ one))),
            float(np.max(np.abs(Dy @ one))),
        )
        if const_defect > self.args.constant_reproduction_tol:
            raise RuntimeError(
                f'MLS constant-derivative defect {const_defect:.3e} exceeds '
                f'{self.args.constant_reproduction_tol:.3e}.'
            )
        print(f'[MLS2] sparse derivative constant defect={const_defect:.3e}')
        return Dx, Dy

    def _raw_geometry_np(self, x: np.ndarray):
        """Return raw F, F^{-1}, J without determinant clipping."""
        x = np.asarray(x, dtype=np.float64)
        F = np.empty((self.N, 2, 2), dtype=np.float64)
        F[:, 0, 0] = self.Dx_sp @ x[:, 0]
        F[:, 0, 1] = self.Dy_sp @ x[:, 0]
        F[:, 1, 0] = self.Dx_sp @ x[:, 1]
        F[:, 1, 1] = self.Dy_sp @ x[:, 1]
        J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        if not np.all(np.isfinite(J)):
            raise RuntimeError('Non-finite raw Jacobian encountered.')
        invF = np.empty_like(F)
        invF[:, 0, 0] = F[:, 1, 1] / J
        invF[:, 0, 1] = -F[:, 0, 1] / J
        invF[:, 1, 0] = -F[:, 1, 0] / J
        invF[:, 1, 1] = F[:, 0, 0] / J
        return F, invF, J

    def _require_admissible_geometry(self, x: np.ndarray, label: str):
        F, invF, J = self._raw_geometry_np(x)
        jmin = float(np.min(J))
        if jmin <= self.args.j_admissible:
            raise RuntimeError(
                f'{label} raw min(J)={jmin:.6e} <= admissible threshold '
                f'{self.args.j_admissible:.6e}.'
            )
        s = np.linalg.svd(F, compute_uv=False)
        cond_max = float(np.max(s[:, 0] / s[:, -1]))
        return F, invF, J, cond_max

    def _assemble_piola_D_sparse(self, x: np.ndarray):
        """All-particle Piola divergence in block velocity ordering [u_x;u_y]."""
        _, invF, J, cond_max = self._require_admissible_geometry(x, 'D configuration')
        Jd = diags(J, format='csr')
        invJd = diags(1.0 / J, format='csr')
        C00 = diags(invF[:, 0, 0], format='csr')
        C01 = diags(invF[:, 0, 1], format='csr')
        C10 = diags(invF[:, 1, 0], format='csr')
        C11 = diags(invF[:, 1, 1], format='csr')
        Dx_u = invJd @ (self.Dx_sp @ Jd @ C00 + self.Dy_sp @ Jd @ C10)
        Dy_u = invJd @ (self.Dx_sp @ Jd @ C01 + self.Dy_sp @ Jd @ C11)
        Dfull = hstack([Dx_u, Dy_u], format='csr')
        return Dfull, J, cond_max

    def _assemble_pressure_G_sparse(self, x: np.ndarray):
        """Pressure action F^{-T} Grad_X/rho in block velocity ordering."""
        _, invF, J, cond_max = self._require_admissible_geometry(x, 'G configuration')
        Dxq = self.Dx_sp[:, self.unknown_idx_np]
        Dyq = self.Dy_sp[:, self.unknown_idx_np]
        Gx = (diags(invF[:, 0, 0]) @ Dxq + diags(invF[:, 1, 0]) @ Dyq) / self.rho0
        Gy = (diags(invF[:, 0, 1]) @ Dxq + diags(invF[:, 1, 1]) @ Dyq) / self.rho0
        G = vstack([Gx, Gy], format='csr')
        return G, J, cond_max

    @staticmethod
    def _block_velocity(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return np.concatenate([v[:, 0], v[:, 1]])

    def _unblock_velocity(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return np.column_stack([v[:self.N], v[self.N:]])

    def initial_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.X_np.copy()
        u = np.zeros_like(x)
        u[:, 0] = self.args.delta0 * x[:, 0]
        u[:, 1] = -self.args.delta0 * x[:, 1]
        p = np.zeros(self.N, dtype=np.float64)
        return x, u, p

    def pack_unknown(self, p_full: np.ndarray) -> np.ndarray:
        return p_full[self.unknown_idx_np].astype(np.float64)

    def unpack_unknown_np(self, p_u: np.ndarray) -> np.ndarray:
        p = np.zeros(self.N, dtype=np.float64)
        p[self.unknown_idx_np] = p_u
        return p

    def make_predictor(self, x_n_np: np.ndarray, u_n_np: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """External-force predictor.

        Returned variables are always the terminal predictor state used by the
        pressure projection:

            u_star = predicted end-of-step velocity before pressure,
            x_star = predicted end-of-step position before pressure.

        For the harmonic external acceleration

            a_ext(x) = -Omega^2 x,

        the option ``implicit-midpoint-force`` solves the midpoint-force
        predictor analytically:

            u_mid = (u_n - 0.5*Omega^2*dt*x_n) / (1 + 0.25*Omega^2*dt^2),
            x_star = x_n + dt*u_mid,
            u_star = 2*u_mid - u_n.

        Thus the force is evaluated consistently at

            x_mid = x_n + 0.5*dt*u_mid,

        rather than explicitly at x_n.
        """
        x_n = jnp.asarray(x_n_np, dtype=self.dtype)
        u_n = jnp.asarray(u_n_np, dtype=self.dtype)

        if self.args.predictor == 'kinematic':
            # Explicit acceleration at x_n.
            a_ext = -self.omega2 * x_n
            u_star = u_n + self.dt * a_ext
            x_star = x_n + self.dt * u_n + 0.5 * self.dt * self.dt * a_ext

        elif self.args.predictor == 'semi-implicit':
            # Explicit acceleration at x_n, then update position with u_star.
            a_ext = -self.omega2 * x_n
            u_star = u_n + self.dt * a_ext
            x_star = x_n + self.dt * u_star

        elif self.args.predictor == 'implicit-midpoint-force':
            # Implicit midpoint for the linear harmonic external force.
            # This is the closed-form solve of
            #   u_mid = u_n - 0.5*Omega^2*dt*(x_n + 0.5*dt*u_mid).
            denom = 1.0 + 0.25 * self.omega2 * self.dt * self.dt
            u_mid = (u_n - 0.5 * self.omega2 * self.dt * x_n) / denom
            x_star = x_n + self.dt * u_mid
            u_star = 2.0 * u_mid - u_n

        else:
            raise ValueError('unknown predictor')
        return x_star, u_star

    def _deriv_X_scalar(self, f: jnp.ndarray) -> jnp.ndarray:
        """Material gradient using polynomial MLS derivative weights."""
        fj = f[self.deriv_neigh]
        gx = jnp.einsum('nk,nk->n', self.deriv_wx * self.deriv_mask, fj)
        gy = jnp.einsum('nk,nk->n', self.deriv_wy * self.deriv_mask, fj)
        return jnp.stack([gx, gy], axis=1)

    def _deriv_X_div(self, q: jnp.ndarray) -> jnp.ndarray:
        """Material divergence using the same polynomial MLS derivative weights."""
        qj = q[self.deriv_neigh]
        divx = jnp.einsum('nk,nk->n', self.deriv_wx * self.deriv_mask, qj[:, :, 0])
        divy = jnp.einsum('nk,nk->n', self.deriv_wy * self.deriv_mask, qj[:, :, 1])
        return divx + divy

    def _mls_F(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # x: (N,2). Return F, invF, J for map X -> x.
        if self.use_mls_derivative_operator:
            gx = self._deriv_X_scalar(x[:, 0])
            gy = self._deriv_X_scalar(x[:, 1])
            F = jnp.stack([
                jnp.stack([gx[:, 0], gx[:, 1]], axis=1),
                jnp.stack([gy[:, 0], gy[:, 1]], axis=1),
            ], axis=1)
        else:
            xj = x[self.neigh]                 # (N,maxn,2)
            xi = x[:, None, :]
            dx = (xj - xi) * self.mask[:, :, None]
            # B_i = sum w dx_ij outer dX_ij
            B = jnp.einsum('nk,nka,nkb->nab', self.wmls * self.mask, dx, self.dX)
            F = jnp.einsum('nab,nbc->nac', B, self.Minv)
        # Mild orientation floor for numerical safety.
        det = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        det_safe = jnp.where(jnp.abs(det) < 1e-8, jnp.sign(det + 1e-16) * 1e-8, det)
        invF = jnp.stack([
            jnp.stack([ F[:, 1, 1] / det_safe, -F[:, 0, 1] / det_safe], axis=1),
            jnp.stack([-F[:, 1, 0] / det_safe,  F[:, 0, 0] / det_safe], axis=1),
        ], axis=1)
        return F, invF, det_safe

    def _grad_X_p(self, p_full: jnp.ndarray) -> jnp.ndarray:
        if self.use_mls_derivative_operator:
            return self._deriv_X_scalar(p_full)
        pj = p_full[self.neigh]
        pi = p_full[:, None]
        dp = (pj - pi) * self.mask
        grad = self.V0 * jnp.einsum('nk,nka->na', dp, self.gradW)
        return grad

    def _div_X_vec(self, q: jnp.ndarray) -> jnp.ndarray:
        if self.use_mls_derivative_operator:
            return self._deriv_X_div(q)
        qj = q[self.neigh]
        qi = q[:, None, :]
        dq = (qj - qi) * self.mask[:, :, None]
        div = self.V0 * jnp.einsum('nka,nka->n', dq, self.gradW)
        return div

    def _unpack_jax(self, p_u: jnp.ndarray) -> jnp.ndarray:
        p = jnp.zeros((self.N,), dtype=self.dtype)
        p = p.at[self.unknown_idx].set(p_u)
        return p

    def _compute_state_from_p(self, p_u: jnp.ndarray, x_n: jnp.ndarray, u_star: jnp.ndarray,
                              x_star: jnp.ndarray, lam: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        p_full = self._unpack_jax(p_u)
        gradp_X = self._grad_X_p(p_full)

        # Fixed point for x_T(p) because x_theta depends on x_T and G depends on x_theta.
        xT = x_star
        theta = self.args.theta
        if self.args.placement == 'star':
            theta = -1.0
        elif self.args.placement == 'current':
            theta = 0.0
        elif self.args.placement == 'terminal':
            theta = 1.0
        elif self.args.placement == 'midpoint':
            theta = 0.5

        def body(_, xT_inner):
            if self.args.placement == 'star':
                x_theta = x_star
            else:
                x_theta = (1.0 - theta) * x_n + theta * xT_inner
            _, invFtheta, _ = self._mls_F(x_theta)
            g = jnp.einsum('nba,nb->na', invFtheta, gradp_X)  # invF^T grad: g_a = invF_{b a} grad_b
            xT_new = x_star - lam * self.args.position_correction * self.dt * self.dt / self.rho0 * g
            return xT_new

        xT = jax.lax.fori_loop(0, self.args.geometry_iters, body, xT)
        if self.args.placement == 'star':
            x_theta = x_star
        else:
            x_theta = (1.0 - theta) * x_n + theta * xT
        _, invFtheta, _ = self._mls_F(x_theta)
        g = jnp.einsum('nba,nb->na', invFtheta, gradp_X)
        uT = u_star - lam * self.dt / self.rho0 * g
        xT = x_star - lam * self.args.position_correction * self.dt * self.dt / self.rho0 * g
        return p_full, gradp_X, g, uT, xT

    def make_residual_fun(self, x_n_np: np.ndarray, u_n_np: np.ndarray):
        x_n = jnp.asarray(x_n_np, dtype=self.dtype)
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)

        def div_piola_at_state(u: jnp.ndarray, xT: jnp.ndarray) -> jnp.ndarray:
            _, invFT, JT = self._mls_F(xT)
            # q = J F^{-1} u
            q = JT[:, None] * jnp.einsum('nab,nb->na', invFT, u)
            div = self._div_X_vec(q) / JT
            return div

        # Homotopy source at lambda=0, p=0.
        source0_full = div_piola_at_state(u_star, x_star)
        source0 = source0_full[self.unknown_idx]

        def residual(p_u: jnp.ndarray, lam: float) -> jnp.ndarray:
            _, _, _, uT, xT = self._compute_state_from_p(p_u, x_n, u_star, x_star, lam)
            div_full = div_piola_at_state(uT, xT)
            r = div_full[self.unknown_idx] - (1.0 - lam) * source0
            return r

        def diagnostics(p_u: jnp.ndarray, lam: float):
            p_full, _, g, uT, xT = self._compute_state_from_p(p_u, x_n, u_star, x_star, lam)
            div_star = source0
            div_full = div_piola_at_state(uT, xT)
            div_u = div_full[self.unknown_idx]
            rms_star = jnp.sqrt(jnp.mean(div_star * div_star) + 1e-30)
            rms_div = jnp.sqrt(jnp.mean(div_u * div_u) + 1e-30)
            max_star = jnp.max(jnp.abs(div_star))
            max_div = jnp.max(jnp.abs(div_u))
            pmax = jnp.max(jnp.abs(p_full))
            gnorm = jnp.sqrt(jnp.sum(g * g, axis=1))
            gmax = jnp.max(gnorm)
            disp = jnp.sqrt(jnp.sum((xT - x_star) ** 2, axis=1))
            dispmax = jnp.max(disp)
            pcenter = p_full[self.center_idx]
            return max_star, rms_star, max_div, rms_div, rms_div / (rms_star + 1e-30), pmax, gmax, dispmax, pcenter, xT, uT, p_full, g

        def residual_jvp(p_u: jnp.ndarray, v: jnp.ndarray, lam: float):
            _, tangent = jax.jvp(lambda pp: residual(pp, lam), (p_u,), (v,))
            return tangent

        return (jax.jit(residual), jax.jit(diagnostics),
                jax.jit(residual_jvp), np.asarray(source0))

    def _solve_nonlinear_one_step(self, step: int, x_n_np: np.ndarray, u_n_np: np.ndarray, p_prev_full: np.ndarray) -> StepResult:
        residual_jit, diag_jit, residual_jvp_jit, source0_np = self.make_residual_fun(x_n_np, u_n_np)
        source_norm = max(1e-30, float(np.linalg.norm(source0_np)))
        # Solver-only frozen D_star G_star preconditioner.  It does not alter the
        # nonlinear residual, accepted state, or configuration assignment.
        x_star_pre, _ = self.make_predictor(x_n_np, u_n_np)
        x_star_pre = np.asarray(x_star_pre, dtype=np.float64)
        Dpre, _, _ = self._assemble_piola_D_sparse(x_star_pre)
        Gpre, _, _ = self._assemble_pressure_G_sparse(x_star_pre)
        Apre = csc_matrix(Dpre[self.unknown_idx_np, :] @ Gpre)
        Apre.sum_duplicates(); Apre.eliminate_zeros()
        try:
            pre_fac = splu(Apre, permc_spec='COLAMD')
        except RuntimeError as exc:
            raise RuntimeError(
                f'Frozen D_star G_star preconditioner is singular at step={step}; '
                'no shift is permitted.'
            ) from exc
        if self.args.pressure_init == 'previous':
            p_u = self.pack_unknown(p_prev_full)
        else:
            p_u = np.zeros(self.n_unknown, dtype=np.float64)

        lam = 0.0
        lam_step = self.args.lambda_step
        total_reductions = 0
        accepted_newton = 0
        last_gmres_info = 0
        last_gmres_it = 0
        last_beta = 0.0
        last_dp_inf = 0.0
        last_dp_rel = 0.0
        last_dg_inf = 0.0
        last_dg_rel = 0.0
        last_rel_before = np.nan
        last_rel_after = np.nan
        last_solve_status = 'init'
        # v11-diagnostic: record why a continuation stage failed.
        last_fail_detail = 'none'
        last_lam_start = 0.0
        last_target_lam = 0.0
        last_lam_step_used = lam_step
        last_rel_full_snap = np.nan
        last_full_snap_tol = np.nan
        last_rel_stage0 = np.nan
        last_stage_tol = np.nan
        last_pre_tol = np.nan
        last_line_reject_detail = 'none'

        def res_np(p_vec: np.ndarray, lam_val: float) -> np.ndarray:
            return np.asarray(residual_jit(jnp.asarray(p_vec, dtype=self.dtype), float(lam_val)), dtype=np.float64)

        t_start = time.time()
        while lam < 1.0 - self.args.lambda_complete_tol:
            # v11 control fix: before taking another continuation stage, check whether
            # the current pressure already solves the full lambda=1 problem. This avoids
            # wasting Newton/GMRES iterations when the full nonlinear residual is already
            # below the requested tolerance, and prevents false failures caused by the
            # finite-difference JVP noise floor.
            if self.args.enable_full_lambda_snap:
                r_full_snap = res_np(p_u, 1.0)
                rel_full_snap = float(np.linalg.norm(r_full_snap) / source_norm)
                full_snap_tol = self.args.full_lambda_snap_rel if self.args.full_lambda_snap_rel > 0.0 else self.args.final_accept_rel
                last_rel_full_snap = rel_full_snap
                last_full_snap_tol = full_snap_tol
                if rel_full_snap <= full_snap_tol:
                    lam = 1.0
                    last_rel_before = rel_full_snap
                    last_rel_after = rel_full_snap
                    last_solve_status = 'full_lambda_snap'
                    break

            target_lam = min(1.0, lam + lam_step)
            last_lam_start = lam
            last_target_lam = target_lam
            last_lam_step_used = lam_step
            p_stage = p_u.copy()
            success_stage = False

            # v11 control fix: if this lambda stage is already sufficiently solved,
            # accept it without forcing the minimum Newton iteration count. The old
            # min-newton rule prevented pressure under-updating, but at very small
            # residuals it can push GMRES into the finite-difference JVP noise floor.
            if self.args.enable_stage_precheck:
                r_stage0 = res_np(p_stage, target_lam)
                rel_stage0 = float(np.linalg.norm(r_stage0) / source_norm)
                stage_tol = self.args.final_accept_rel if target_lam >= 1.0 - self.args.lambda_complete_tol else self.args.lambda_accept_rel
                pre_tol = self.args.stage_precheck_rel if self.args.stage_precheck_rel > 0.0 else stage_tol
                last_rel_stage0 = rel_stage0
                last_stage_tol = stage_tol
                last_pre_tol = pre_tol
                if rel_stage0 <= min(stage_tol, pre_tol):
                    success_stage = True
                    last_rel_before = rel_stage0
                    last_rel_after = rel_stage0
                    last_solve_status = 'stage_precheck_converged'

            if not success_stage:
                for nit in range(self.args.newton_iters):
                    r0 = res_np(p_stage, target_lam)
                    source_scale = source_norm
                    rel_now = float(np.linalg.norm(r0) / source_scale)
                    dvals = diag_jit(jnp.asarray(p_stage, dtype=self.dtype), float(target_lam))
                    # Use a possibly looser final tolerance at lambda=1, and stage tolerance otherwise.
                    accept_tol = self.args.final_accept_rel if target_lam >= 1.0 - self.args.lambda_complete_tol else self.args.lambda_accept_rel
                    last_rel_before = rel_now
                    # Do not skip pressure solve solely because divergence is small.
                    # Pressure is p^{n+theta}; it must also pass an update/convergence check.
                    # Therefore at least --min-newton-iters Newton/JVP attempts are made
                    # before accepting initial residual convergence.
                    if rel_now <= accept_tol and nit >= self.args.min_newton_iters:
                        success_stage = True
                        last_solve_status = 'residual_converged_after_min_iters'
                        break

                    pnorm = np.linalg.norm(p_stage)
                    eps_base = self.args.jvp_eps * (1.0 + pnorm)

                    gmres_counter = {'n': 0}

                    def matvec(v: np.ndarray) -> np.ndarray:
                        nv = np.linalg.norm(v)
                        if nv < 1e-300:
                            return np.zeros_like(v)
                        if self.args.jvp_mode == 'autodiff':
                            jv = np.asarray(residual_jvp_jit(
                                jnp.asarray(p_stage, dtype=self.dtype),
                                jnp.asarray(v, dtype=self.dtype),
                                float(target_lam),
                            ), dtype=np.float64)
                            return jv + self.args.newton_damp * v
                        eps = eps_base / nv
                        return ((res_np(p_stage + eps * v, target_lam) - r0) / eps
                                + self.args.newton_damp * v)

                    Aop = LinearOperator((self.n_unknown, self.n_unknown), matvec=matvec, dtype=np.float64)

                    # J approximately equals -target_lam*dt*(D_star G_star).
                    pre_scale = -1.0 / max(float(target_lam) * self.dt, 1e-300)
                    Mop = LinearOperator(
                        (self.n_unknown, self.n_unknown),
                        matvec=lambda vv: pre_scale * pre_fac.solve(vv),
                        dtype=np.float64,
                    )

                    def cb(_):
                        gmres_counter['n'] += 1

                    if self.args.krylov == 'lgmres':
                        delta, info = lgmres(Aop, -r0, rtol=self.args.gmres_tol, atol=0.0,
                                             M=Mop, maxiter=self.args.gmres_iters, callback=cb)
                    else:
                        delta, info = gmres(Aop, -r0, M=Mop, rtol=self.args.gmres_tol, atol=0.0,
                                            restart=self.args.gmres_restart, maxiter=self.args.gmres_iters,
                                            callback=cb, callback_type='legacy')
                    last_gmres_info = int(info) if isinstance(info, (int, np.integer)) else -999
                    last_gmres_it = gmres_counter['n']

                    # Trust-region on the Newton pressure increment.  This is not a
                    # pressure-branch selector; it only prevents a noisy/non-converged
                    # Krylov direction from jumping to a remote pressure state.
                    dmax = float(np.max(np.abs(delta))) if delta.size else 0.0
                    pscale = max(1.0, float(np.max(np.abs(p_stage))) if p_stage.size else 0.0)
                    dlimit = max(self.args.max_delta_p_abs, self.args.max_delta_p_frac * pscale)
                    if dmax > dlimit and dmax > 0.0:
                        delta = delta * (dlimit / dmax)

                    # Current pressure force for update-convergence diagnostics.
                    dvals_stage = diag_jit(jnp.asarray(p_stage, dtype=self.dtype), float(target_lam))
                    g_stage = np.asarray(dvals_stage[12], dtype=np.float64)
                    g_stage_norm = np.sqrt(np.sum(g_stage * g_stage, axis=1))
                    g_stage_scale = max(1.0, float(np.max(g_stage_norm)))
                    p_stage_scale = max(1.0, float(np.max(np.abs(p_stage))) if p_stage.size else 0.0)

                    accepted = False
                    best_p = p_stage
                    best_rel = rel_now
                    best_dp_inf = 0.0
                    best_dp_rel = 0.0
                    best_dg_inf = 0.0
                    best_dg_rel = 0.0
                    reject_gmres_beta = 0
                    reject_nonfinite = 0
                    reject_disp = 0
                    reject_no_improve = 0
                    tested_beta = 0
                    best_trial_rel_seen = np.inf
                    best_trial_beta_seen = 0.0
                    best_trial_disp_seen = np.nan
                    for beta in self.beta_grid:
                        if last_gmres_info != 0 and abs(beta) > self.args.max_beta_if_gmres_fail:
                            reject_gmres_beta += 1
                            continue
                        cand = p_stage + beta * delta
                        r_c = res_np(cand, target_lam)
                        rel_c = float(np.linalg.norm(r_c) / source_scale)
                        dvals_c = diag_jit(jnp.asarray(cand, dtype=self.dtype), float(target_lam))
                        dispmax = float(np.asarray(dvals_c[7]))
                        pmax = float(np.asarray(dvals_c[5]))
                        gmax = float(np.asarray(dvals_c[6]))
                        g_c = np.asarray(dvals_c[12], dtype=np.float64)
                        step_dp = beta * delta
                        dp_inf_c = float(np.max(np.abs(step_dp))) if step_dp.size else 0.0
                        dp_rel_c = dp_inf_c / p_stage_scale
                        dg_vec = g_c - g_stage
                        dg_norm = np.sqrt(np.sum(dg_vec * dg_vec, axis=1))
                        dg_inf_c = float(np.max(dg_norm)) if dg_norm.size else 0.0
                        dg_rel_c = dg_inf_c / g_stage_scale
                        if beta == 0.0:
                            continue
                        tested_beta += 1
                        if rel_c < best_trial_rel_seen:
                            best_trial_rel_seen = rel_c
                            best_trial_beta_seen = beta
                            best_trial_disp_seen = dispmax
                        if not np.isfinite(rel_c + dispmax + pmax + gmax):
                            reject_nonfinite += 1
                            continue
                        if dispmax > self.args.max_pressure_displacement_frac * self.dx:
                            reject_disp += 1
                            continue
                        # True nonlinear residual must decrease.
                        if rel_c < best_rel * (1.0 - self.args.min_improve):
                            best_rel = rel_c
                            best_p = cand
                            best_dp_inf = dp_inf_c
                            best_dp_rel = dg_rel_c if False else dp_rel_c
                            best_dg_inf = dg_inf_c
                            best_dg_rel = dg_rel_c
                            accepted = True
                            last_beta = beta
                            break
                        else:
                            reject_no_improve += 1
                    last_line_reject_detail = (
                        f"tested_beta={tested_beta}, reject_gmres_beta={reject_gmres_beta}, "
                        f"reject_nonfinite={reject_nonfinite}, reject_disp={reject_disp}, "
                        f"reject_no_improve={reject_no_improve}, best_trial_beta={best_trial_beta_seen:.3g}, "
                        f"best_trial_rel={best_trial_rel_seen:.3e}, best_trial_disp={best_trial_disp_seen:.3e}, "
                        f"required_rel<{best_rel * (1.0 - self.args.min_improve):.3e}"
                    )
                    if not accepted:
                        # If a minimum number of pressure attempts has been made and
                        # the residual is already acceptable, declare convergence.
                        # Otherwise, this stage has failed and lambda/dt must be reduced.
                        if rel_now <= accept_tol and nit >= self.args.min_newton_iters:
                            success_stage = True
                            last_solve_status = 'no_accepted_step_but_residual_ok'
                        else:
                            last_solve_status = 'line_search_failed'
                        break
                    p_stage = best_p
                    accepted_newton += 1
                    last_dp_inf = best_dp_inf
                    last_dp_rel = best_dp_rel
                    last_dg_inf = best_dg_inf
                    last_dg_rel = best_dg_rel
                    last_rel_after = best_rel
                    last_solve_status = 'accepted_newton'
                    accept_tol = self.args.final_accept_rel if target_lam >= 1.0 - self.args.lambda_complete_tol else self.args.lambda_accept_rel
                    pressure_update_small = (best_dp_inf <= self.args.pressure_update_abs_tol or best_dp_rel <= self.args.pressure_update_rel_tol)
                    force_update_small = (best_dg_inf <= self.args.force_update_abs_tol or best_dg_rel <= self.args.force_update_rel_tol)
                    if best_rel <= accept_tol and (nit + 1) >= self.args.min_newton_iters and pressure_update_small and force_update_small:
                        success_stage = True
                        last_solve_status = 'residual_and_update_converged'
                        break

            if success_stage:
                p_u = p_stage
                lam = target_lam
                lam_step = min(self.args.lambda_step_max, lam_step * self.args.lambda_growth)
            else:
                last_fail_detail = (
                    f"stage_failed: status={last_solve_status}, lam_start={last_lam_start:.6f}, "
                    f"target_lam={last_target_lam:.6f}, lam_step_used={last_lam_step_used:.3e}, "
                    f"rel_full_snap={last_rel_full_snap:.3e}/tol={last_full_snap_tol:.3e}, "
                    f"rel_stage0={last_rel_stage0:.3e}, stage_tol={last_stage_tol:.3e}, pre_tol={last_pre_tol:.3e}, "
                    f"rel_before={last_rel_before:.3e}, rel_after={last_rel_after:.3e}, "
                    f"gmres_it={last_gmres_it}, gmres_info={last_gmres_info}, beta={last_beta:.3g}, "
                    f"dp_inf={last_dp_inf:.3e}, dg_inf={last_dg_inf:.3e}; {last_line_reject_detail}"
                )
                lam_step *= 0.5
                total_reductions += 1
                if lam_step < self.args.lambda_step_min:
                    # Strict continuation: do NOT advance a physical time step with lambda < 1.
                    # A partial lambda solution only solves an auxiliary homotopy problem, not the
                    # real terminal incompressibility equation.  The caller can choose to abort
                    # or explicitly allow this diagnostic behavior.
                    if self.args.allow_incomplete_step:
                        # Diagnostic mode only: accept the best available partial-stage pressure.
                        p_u = p_stage
                        lam = target_lam
                    break

        if lam >= 1.0 - self.args.lambda_complete_tol:
            lam = 1.0
        # v11 final rescue: even if staged continuation did not explicitly walk to
        # lambda=1, the current pressure may already solve the full problem. This
        # happens when the residual is below the finite-difference JVP noise floor and
        # forcing additional GMRES iterations creates false continuation failures.
        if lam < 1.0 - self.args.lambda_complete_tol and self.args.enable_full_lambda_snap:
            r_full_final = res_np(p_u, 1.0)
            rel_full_final = float(np.linalg.norm(r_full_final) / source_norm)
            full_snap_tol = self.args.full_lambda_snap_rel if self.args.full_lambda_snap_rel > 0.0 else self.args.final_accept_rel
            if rel_full_final <= full_snap_tol:
                lam = 1.0
                last_rel_after = rel_full_final
                last_solve_status = 'full_lambda_snap_final'
        if lam < 1.0 - self.args.lambda_complete_tol and not self.args.allow_incomplete_step:
            rel_here = float(np.linalg.norm(res_np(p_u, lam)) / source_norm) if lam > 0 else 1.0
            raise RuntimeError(
                f"Continuation failed at physical step {step}: reached lambda={lam:.6f} < 1. "
                f"Last accepted relative residual={rel_here:.3e}. "
                f"Internal failure detail: {last_fail_detail}. "
                f"Use smaller --dt, smaller --lambda-step, larger --newton-iters/--gmres-iters, "
                f"or pass --allow-incomplete-step only for diagnostics."
            )

        dvals = diag_jit(jnp.asarray(p_u, dtype=self.dtype), 1.0)
        d_np = [np.asarray(v) for v in dvals]
        xT = np.asarray(d_np[9], dtype=np.float64)
        uT = np.asarray(d_np[10], dtype=np.float64)
        p_full = np.asarray(d_np[11], dtype=np.float64)
        max_star, rms_star, max_div, rms_div, rel, pmax, gmax, dispmax, pcenter = [float(v) for v in d_np[:9]]

        diag = {
            'max_D_star': max_star,
            'rms_D_star': rms_star,
            'max_D_np1': max_div,
            'rms_D_np1': rms_div,
            'rel_rms': rel,
            'pmax': pmax,
            'gmax': gmax,
            'dispmax': dispmax,
            'p_center': pcenter,
            'lam': lam,
            'lambda_reductions': float(total_reductions),
            'accepted_newton': float(accepted_newton),
            'gmres_info': float(last_gmres_info),
            'gmres_it': float(last_gmres_it),
            'beta': float(last_beta),
            'pressure_time_theta': self.effective_pressure_theta(),
            'dp_inf': float(last_dp_inf),
            'dp_rel': float(last_dp_rel),
            'dg_inf': float(last_dg_inf),
            'dg_rel': float(last_dg_rel),
            'rel_before': float(last_rel_before) if np.isfinite(last_rel_before) else float('nan'),
            'rel_after': float(last_rel_after) if np.isfinite(last_rel_after) else float('nan'),
            'solve_status_code': float({
                'init': 0,
                'accepted_newton': 1,
                'residual_converged_after_min_iters': 2,
                'no_accepted_step_but_residual_ok': 3,
                'line_search_failed': 4,
                'residual_and_update_converged': 5,
                'stage_precheck_converged': 6,
                'full_lambda_snap': 7,
                'full_lambda_snap_final': 8,
            }.get(last_solve_status, -1)),
            'wall_time': time.time() - t_start,
        }
        diag['p_center_fit'] = self.center_pressure_fit(xT, p_full)
        return StepResult(xT, uT, p_full, diag)

    def _pressure_work_diagnostics(self, u_star: np.ndarray, uT: np.ndarray,
                                   g: np.ndarray):
        mass = self.rho0 * self.V0
        delta_k = 0.5 * mass * float(np.sum(uT*uT) - np.sum(u_star*u_star))
        work = (-self.dt * mass * float(np.sum(u_star*g))
                + 0.5 * self.dt*self.dt * mass * float(np.sum(g*g)))
        return delta_k, work, abs(delta_k-work)

    def _polygon_area(self, x: np.ndarray) -> float:
        idx = np.asarray(self.material_free_idx_np, dtype=np.int64)
        if idx.size < 3:
            return float('nan')
        xb = np.asarray(x, dtype=np.float64)[idx]
        c = np.mean(xb, axis=0)
        order = np.argsort(np.arctan2(xb[:, 1]-c[1], xb[:, 0]-c[0]))
        xb = xb[order]
        return 0.5 * abs(float(np.sum(
            xb[:, 0] * np.roll(xb[:, 1], -1)
            - xb[:, 1] * np.roll(xb[:, 0], -1)
        )))

    def _hard_audit_nonlinear(self, x_n_np: np.ndarray, u_n_np: np.ndarray,
                              result: StepResult) -> StepResult:
        """Independent raw-J rebuild for the accepted D_T/G_theta state."""
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = np.asarray(x_star, dtype=np.float64)
        u_star = np.asarray(u_star, dtype=np.float64)
        xT = np.asarray(result.x, dtype=np.float64)
        uT = np.asarray(result.u, dtype=np.float64)
        p_full = np.asarray(result.p_full, dtype=np.float64)

        if self.branch == 'ccop-midpoint':
            x_action = 0.5 * (np.asarray(x_n_np, dtype=np.float64) + xT)
        elif self.branch == 'terminal-terminal':
            x_action = xT
        else:
            raise RuntimeError(f'Unexpected nonlinear branch {self.branch!r}.')

        _, invFa, Ja, cond_a = self._require_admissible_geometry(x_action, 'accepted action')
        Dfull_T, JT, cond_T = self._assemble_piola_D_sparse(xT)
        Dq_T = Dfull_T[self.unknown_idx_np, :]
        dT = np.asarray(Dq_T @ self._block_velocity(uT)).ravel()
        Dfull_star, Jstar, _ = self._assemble_piola_D_sparse(x_star)
        source = np.asarray(
            Dfull_star[self.unknown_idx_np, :] @ self._block_velocity(u_star)
        ).ravel()
        rel_hard = float(np.linalg.norm(dT) / max(np.linalg.norm(source), 1e-30))

        gradX = np.column_stack([self.Dx_sp @ p_full, self.Dy_sp @ p_full])
        g = np.empty_like(gradX)
        g[:, 0] = (invFa[:, 0, 0]*gradX[:, 0] + invFa[:, 1, 0]*gradX[:, 1]) / self.rho0
        g[:, 1] = (invFa[:, 0, 1]*gradX[:, 0] + invFa[:, 1, 1]*gradX[:, 1]) / self.rho0
        x_check = x_star - self.args.position_correction * self.dt*self.dt * g
        geometry_hard = float(np.sqrt(np.mean((x_check-xT)**2)) / self.dx)

        if rel_hard > self.args.final_accept_rel * (1.0 + self.args.accept_rel_slack):
            raise RuntimeError(
                f'Hard terminal residual failed: rel={rel_hard:.3e} > '
                f'{self.args.final_accept_rel:.3e}.'
            )
        if geometry_hard > self.args.geometry_accept_rel:
            raise RuntimeError(
                f'Hard geometry residual failed: rms/dx={geometry_hard:.3e} > '
                f'{self.args.geometry_accept_rel:.3e}.'
            )

        delta_k, work, work_defect = self._pressure_work_diagnostics(u_star, uT, g)
        d = dict(result.diag)
        d.update({
            'branch': self.branch,
            'rms_D_anchor_after': float(np.sqrt(np.mean(dT*dT)+1e-30)),
            'max_D_anchor_after': float(np.max(np.abs(dT))),
            'rms_D_terminal': float(np.sqrt(np.mean(dT*dT)+1e-30)),
            'max_D_terminal': float(np.max(np.abs(dT))),
            'hard_resid_rel': rel_hard,
            'geometry_resid_rel_hard': geometry_hard,
            'min_J_star': float(np.min(Jstar)),
            'min_J_action': float(np.min(Ja)),
            'min_J_terminal': float(np.min(JT)),
            'max_J_terminal': float(np.max(JT)),
            'max_cond_F_action': cond_a,
            'max_cond_F_terminal': cond_T,
            'represented_area': float(self.V0*np.sum(JT)),
            'polygon_area': self._polygon_area(xT),
            'delta_K_pressure': delta_k,
            'pressure_work_identity': work,
            'pressure_work_identity_defect': work_defect,
            'linear_resid_rel': float('nan'),
        })
        return StepResult(xT, uT, p_full, d)

    def _solve_star_star_one_step(self, step: int, x_n_np: np.ndarray,
                                  u_n_np: np.ndarray,
                                  p_prev_full: np.ndarray) -> StepResult:
        """Matched frozen predictor projection D_star^q G_star."""
        t0 = time.time()
        x_star, u_star = self.make_predictor(x_n_np, u_n_np)
        x_star = np.asarray(x_star, dtype=np.float64)
        u_star = np.asarray(u_star, dtype=np.float64)
        Dfull_star, Jstar, cond_star = self._assemble_piola_D_sparse(x_star)
        Gstar, Jg, cond_g = self._assemble_pressure_G_sparse(x_star)
        Dq_star = Dfull_star[self.unknown_idx_np, :]
        A = csc_matrix(Dq_star @ Gstar)
        A.sum_duplicates()
        A.eliminate_zeros()
        source = np.asarray(Dq_star @ self._block_velocity(u_star)).ravel()
        rhs = source / self.dt
        rhs_norm = max(float(np.linalg.norm(rhs)), 1e-30)
        try:
            fac = splu(A, permc_spec='COLAMD')
        except RuntimeError as exc:
            raise RuntimeError(
                f'star-star pressure matrix is singular at step={step}; no shift or '
                f'pseudo-inverse is permitted: {exc}'
            ) from exc
        piv = np.abs(fac.U.diagonal())
        pivot_ratio = float(np.min(piv) / max(np.max(piv), 1e-300))
        if not np.all(np.isfinite(piv)) or pivot_ratio <= self.args.lu_pivot_ratio_min:
            raise RuntimeError(
                f'star-star pressure matrix failed pivot audit at step={step}: '
                f'min/max |U_ii|={pivot_ratio:.3e}.'
            )
        p_u = np.asarray(fac.solve(rhs), dtype=np.float64)
        lin_rel = float(np.linalg.norm(A @ p_u-rhs) / rhs_norm)
        if lin_rel > self.args.linear_accept_rel:
            raise RuntimeError(
                f'star-star pressure residual failed at step={step}: '
                f'{lin_rel:.3e} > {self.args.linear_accept_rel:.3e}.'
            )

        g = self._unblock_velocity(np.asarray(Gstar @ p_u).ravel())
        uT = u_star - self.dt*g
        xT = x_star - self.args.position_correction*self.dt*self.dt*g
        _, _, JT, cond_T = self._require_admissible_geometry(xT, 'star-star terminal')
        Dfull_T, _, _ = self._assemble_piola_D_sparse(xT)
        d_anchor = np.asarray(Dq_star @ self._block_velocity(uT)).ravel()
        d_terminal = np.asarray(
            Dfull_T[self.unknown_idx_np, :] @ self._block_velocity(uT)
        ).ravel()
        anchor_rel = float(np.linalg.norm(d_anchor) / max(np.linalg.norm(source), 1e-30))
        if anchor_rel > self.args.linear_accept_rel:
            raise RuntimeError(
                f'star-star anchor closure failed at step={step}: '
                f'{anchor_rel:.3e} > {self.args.linear_accept_rel:.3e}.'
            )

        p_full = np.zeros(self.N, dtype=np.float64)
        p_full[self.unknown_idx_np] = p_u
        p_prev_u = np.asarray(p_prev_full, dtype=np.float64)[self.unknown_idx_np]
        g_prev = self._unblock_velocity(np.asarray(Gstar @ p_prev_u).ravel())
        dp_inf = float(np.max(np.abs(p_u-p_prev_u)))
        dp_rel = dp_inf / max(1.0, float(np.max(np.abs(p_prev_u))))
        dg_inf = float(np.max(np.linalg.norm(g-g_prev, axis=1)))
        dg_rel = dg_inf / max(1.0, float(np.max(np.linalg.norm(g_prev, axis=1))))
        delta_k, work, work_defect = self._pressure_work_diagnostics(u_star, uT, g)
        disp = np.linalg.norm(xT-x_star, axis=1)
        gnorm = np.linalg.norm(g, axis=1)
        d = {
            'branch': self.branch,
            'max_D_star': float(np.max(np.abs(source))),
            'rms_D_star': float(np.sqrt(np.mean(source*source)+1e-30)),
            'max_D_np1': float(np.max(np.abs(d_anchor))),
            'rms_D_np1': float(np.sqrt(np.mean(d_anchor*d_anchor)+1e-30)),
            'rms_D_anchor_after': float(np.sqrt(np.mean(d_anchor*d_anchor)+1e-30)),
            'max_D_anchor_after': float(np.max(np.abs(d_anchor))),
            'rms_D_terminal': float(np.sqrt(np.mean(d_terminal*d_terminal)+1e-30)),
            'max_D_terminal': float(np.max(np.abs(d_terminal))),
            'rel_rms': anchor_rel,
            'hard_resid_rel': anchor_rel,
            'linear_resid_rel': lin_rel,
            'geometry_resid_rel_hard': 0.0,
            'pmax': float(np.max(np.abs(p_full))),
            'gmax': float(np.max(gnorm)),
            'dispmax': float(np.max(disp)),
            'p_center': float(p_full[self.center_idx]),
            'p_center_fit': self.center_pressure_fit(xT, p_full),
            'lam': 1.0,
            'lambda_reductions': 0,
            'accepted_newton': 0,
            'gmres_it': 1,
            'gmres_info': 0,
            'beta': 1.0,
            'dp_inf': dp_inf,
            'dp_rel': dp_rel,
            'dg_inf': dg_inf,
            'dg_rel': dg_rel,
            'solve_status_code': 0,
            'wall_time': time.time()-t0,
            'min_J_star': float(np.min(Jstar)),
            'min_J_action': float(np.min(Jg)),
            'min_J_terminal': float(np.min(JT)),
            'max_J_terminal': float(np.max(JT)),
            'max_cond_F_action': max(cond_star, cond_g),
            'max_cond_F_terminal': cond_T,
            'represented_area': float(self.V0*np.sum(JT)),
            'polygon_area': self._polygon_area(xT),
            'delta_K_pressure': delta_k,
            'pressure_work_identity': work,
            'pressure_work_identity_defect': work_defect,
            'lu_pivot_ratio': pivot_ratio,
        }
        return StepResult(xT, uT, p_full, d)

    def solve_one_step(self, step: int, x_n_np: np.ndarray, u_n_np: np.ndarray,
                       p_prev_full: np.ndarray) -> StepResult:
        if self.branch == 'star-star':
            return self._solve_star_star_one_step(step, x_n_np, u_n_np, p_prev_full)
        try:
            result = self._solve_nonlinear_one_step(step, x_n_np, u_n_np, p_prev_full)
        except RuntimeError as exc:
            # Preserve the requested nominal dt: a failed branch is rejected rather
            # than silently compared after branch-dependent adaptive substepping.
            raise RuntimeError(
                f'Strict nominal-dt nonlinear solve rejected at step={step}: {exc}'
            ) from exc
        return self._hard_audit_nonlinear(x_n_np, u_n_np, result)

    def center_pressure_fit(self, x: np.ndarray, p: np.ndarray, radius_factor: float = None) -> float:
        """Quadratic least-squares extrapolation of pressure to x=0.

        This is a diagnostic only.  It reduces sensitivity to a single center
        particle if the reference operator has local ringing near the origin.
        Fit basis: [1, x, y, x^2, xy, y^2] inside r <= radius_factor*R.
        """
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

    def compute_energy(self, x: np.ndarray, u: np.ndarray) -> Tuple[float, float, float]:
        K = 0.5 * self.rho0 * self.V0 * float(np.sum(u * u))
        V = 0.5 * self.rho0 * self.V0 * self.omega2 * float(np.sum(x * x))
        return K, V, K + V

    def run(self):
        x, u, p = self.initial_state()
        xs, us, ps, times = [], [], [], []
        center_p, center_pref = [], []
        rels, pmaxs, gmaxs = [], [], []
        dp_infs, dp_rels, dg_infs, dg_rels = [], [], [], []
        anchor_rms, terminal_rms, terminal_max = [], [], []
        hard_rels, geom_hard, linear_rels = [], [], []
        min_J_star, min_J_action, min_J_terminal, max_J_terminal = [], [], [], []
        cond_F_action, cond_F_terminal = [], []
        represented_areas, polygon_areas = [], []
        delta_K_pressure, pressure_work, pressure_work_defect = [], [], []
        wall_times = []
        center_p_fit = []
        substeps_saved = []
        dt_last_saved = []
        t_p_saved = []

        dt_nominal = self.dt_nominal
        t_now = 0.0
        dt_sub_next = dt_nominal * self.args.dt_substep_init_factor
        dt_min = dt_nominal * self.args.dt_min_factor

        K, V, E = self.compute_energy(x, u)
        print(f"step=00000, t=0.000000e+00, K={K:.8e}, V={V:.8e}, E={E:.8e}, Piola diagnostics: N/A")
        xs.append(x.copy()); us.append(u.copy()); ps.append(p.copy()); times.append(0.0)
        center_p.append(p[self.center_idx]); center_pref.append(reference_center_pressure(0.0, self.R, self.args.delta0, self.args.omega, self.rho0)); center_p_fit.append(self.center_pressure_fit(x, p))
        rels.append(np.nan); pmaxs.append(np.nan); gmaxs.append(np.nan)
        dp_infs.append(np.nan); dp_rels.append(np.nan); dg_infs.append(np.nan); dg_rels.append(np.nan)
        anchor_rms.append(np.nan); terminal_rms.append(np.nan); terminal_max.append(np.nan)
        hard_rels.append(np.nan); geom_hard.append(np.nan); linear_rels.append(np.nan)
        min_J_star.append(1.0); min_J_action.append(1.0); min_J_terminal.append(1.0); max_J_terminal.append(1.0)
        cond_F_action.append(1.0); cond_F_terminal.append(1.0)
        represented_areas.append(float(self.V0*self.N)); polygon_areas.append(self._polygon_area(x))
        delta_K_pressure.append(0.0); pressure_work.append(0.0); pressure_work_defect.append(0.0)
        wall_times.append(0.0)
        substeps_saved.append(0); dt_last_saved.append(np.nan); t_p_saved.append(0.0)

        def write_output_checkpoint(tag='latest'):
            """Write the accumulated saved frames immediately.

            This makes the output npz usable even if the nonlinear solve fails later.
            The file self.args.output is overwritten as a rolling checkpoint.
            """
            np.savez(
                self.args.output,
                x=np.asarray(xs),
                u=np.asarray(us),
                p=np.asarray(ps),
                times=np.asarray(times),
                center_pressure=np.asarray(center_p),
                center_pressure_fit=np.asarray(center_p_fit),
                center_pressure_ref=np.asarray(center_pref),
                rel_rms=np.asarray(rels),
                pmax=np.asarray(pmaxs),
                gmax=np.asarray(gmaxs),
                dp_inf=np.asarray(dp_infs),
                dp_rel=np.asarray(dp_rels),
                dg_inf=np.asarray(dg_infs),
                dg_rel=np.asarray(dg_rels),
                rms_D_anchor_after=np.asarray(anchor_rms),
                rms_D_terminal=np.asarray(terminal_rms),
                max_D_terminal=np.asarray(terminal_max),
                hard_resid_rel=np.asarray(hard_rels),
                geometry_resid_rel_hard=np.asarray(geom_hard),
                linear_resid_rel=np.asarray(linear_rels),
                min_J_star=np.asarray(min_J_star),
                min_J_action=np.asarray(min_J_action),
                min_J_terminal=np.asarray(min_J_terminal),
                max_J_terminal=np.asarray(max_J_terminal),
                max_cond_F_action=np.asarray(cond_F_action),
                max_cond_F_terminal=np.asarray(cond_F_terminal),
                represented_area=np.asarray(represented_areas),
                polygon_area=np.asarray(polygon_areas),
                delta_K_pressure=np.asarray(delta_K_pressure),
                pressure_work_identity=np.asarray(pressure_work),
                pressure_work_identity_defect=np.asarray(pressure_work_defect),
                wall_time_per_saved_step=np.asarray(wall_times),
                substeps=np.asarray(substeps_saved),
                dt_last=np.asarray(dt_last_saved),
                pressure_times=np.asarray(t_p_saved),
                X=self.X_np,
                unknown_idx=self.unknown_idx_np,
                free_idx=self.free_idx_np,
                center_idx=np.array([self.center_idx]),
                dx=np.array([self.dx]),
                h=np.array([self.h]),
                rho0=np.array([self.rho0]),
                dt=np.array([dt_nominal]),
                radius=np.array([self.R]),
                delta0=np.array([self.args.delta0]),
                omega=np.array([self.args.omega]),
                pressure_time_theta=np.array([self.effective_pressure_theta()]),
                branch=np.array([self.branch]),
                final_time_requested=np.array([self.args.steps*dt_nominal]),
                R_over_dx=np.array([self.R/self.dx]),
                last_checkpoint_tag=np.array([str(tag)]),
            )


        # Time-roundoff tolerance for hitting each nominal output time.
        # This is deliberately independent from dt_min.  dt_min controls nonlinear
        # rejection reductions, not the harmless final floating-point remainder.
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
                    # Snap away accumulated roundoff such as 1e-14 remainders.
                    t_now = target_time
                    break

                # dt_sub_next is the proposed nonlinear substep.  It is not allowed
                # to go below dt_min when there is still a real chunk of the physical
                # step left.  However, a final remainder smaller than dt_min is allowed
                # because it is easier than dt_min and only exists to land exactly on
                # target_time.
                if dt_sub_next < dt_min - 10.0 * np.finfo(float).eps * dt_nominal and remaining > dt_min:
                    write_output_checkpoint(tag=f'failure_step_{step:05d}')
                    print('failure checkpoint saved:', self.args.output)
                    raise RuntimeError(
                        f"Adaptive substep failed before physical step {step}: proposed dt_sub_next={dt_sub_next:.3e} < dt_min={dt_min:.3e}. "
                        f"Try smaller --dt, lower --dt-min-factor, larger --newton-iters/--gmres-iters, or better preconditioning."
                    )

                dt_try = min(dt_sub_next, remaining)
                if dt_try <= 0.0:
                    raise RuntimeError(
                        f"Non-positive adaptive substep at physical step {step}: dt_try={dt_try:.3e}, remaining={remaining:.3e}."
                    )

                if n_sub >= self.args.max_substeps_per_step:
                    write_output_checkpoint(tag=f'failure_step_{step:05d}')
                    print('failure checkpoint saved:', self.args.output)
                    raise RuntimeError(
                        f"Exceeded --max-substeps-per-step={self.args.max_substeps_per_step} at physical step {step}. "
                        f"Last dt_try={dt_try:.3e}, remaining={remaining:.3e}."
                    )

                old_dt = self.dt
                self.dt = float(dt_try)
                try:
                    result = self.solve_one_step(step, x, u, p)
                except RuntimeError as e:
                    self.dt = old_dt
                    msg = str(e)
                    if 'Continuation failed' not in msg:
                        write_output_checkpoint(tag=f'failure_step_{step:05d}')
                        print('failure checkpoint saved:', self.args.output)
                        raise
                    # If the actual attempted nonlinear substep is already at the
                    # minimum and it still fails, report a true nonlinear failure.
                    if dt_try <= dt_min * (1.0 + 1e-10) and remaining > dt_min:
                        write_output_checkpoint(tag=f'failure_step_{step:05d}')
                        print('failure checkpoint saved:', self.args.output)
                        raise RuntimeError(
                            f"Adaptive substep reached dt_min but continuation still failed at physical step {step}: "
                            f"dt_try={dt_try:.3e}, dt_min={dt_min:.3e}. Original reason: {msg.splitlines()[0]}"
                        ) from e
                    dt_sub_next = max(0.5 * dt_try, dt_min)
                    print(
                        f"[adaptive-substep] reject step={step:05d}, t={t_now:.8e}, "
                        f"dt={dt_try:.3e}; retry dt={dt_sub_next:.3e}; reason: {msg.splitlines()[0]}"
                    )
                    continue
                finally:
                    self.dt = old_dt

                # Accept substep only if solve_one_step completed with lambda=1.
                x, u, p = result.x, result.u, result.p_full
                last_result = result
                last_dt = dt_try
                last_t_p = t_now + self.effective_pressure_theta() * dt_try
                t_now += dt_try
                n_sub += 1
                dt_sub_next = min(dt_nominal, max(dt_min, dt_try * self.args.dt_substep_growth))

            K, V, E = self.compute_energy(x, u)
            if last_result is None:
                write_output_checkpoint(tag=f'failure_step_{step:05d}')
                print('failure checkpoint saved:', self.args.output)
                raise RuntimeError(f"No accepted substep for physical step {step}")
            pref = reference_center_pressure(last_t_p, self.R, self.args.delta0, self.args.omega, self.rho0)
            if step == 1 or step % self.args.save_every == 0:
                d = last_result.diag
                print(
                    f"step={step:05d}, t={t_now:.8e}, K={K:.8e}, V={V:.8e}, E={E:.8e}, "
                    f"predictor-source-rms={d['rms_D_star']:.3e}, "
                    f"rel_rms={d['rel_rms']:.3e}, lam={d['lam']:.3f}, red={int(d['lambda_reductions'])}, "
                    f"newton_acc={int(d['accepted_newton'])}, gmres_it={int(d['gmres_it'])}, gmres_info={int(d['gmres_info'])}, "
                    f"beta={d['beta']:.3f}, dispmax={d['dispmax']:.3e}, "
                    f"p_center={d['p_center']:.6e}, p_center_fit={d['p_center_fit']:.6e}, p_center_ref={pref:.6e}, t_p={last_t_p:.6e}, "
                    f"pmax={d['pmax']:.3e}, gmax={d['gmax']:.3e}, "
                    f"dp_inf={d['dp_inf']:.3e}, dg_inf={d['dg_inf']:.3e}, status={int(d['solve_status_code'])}, "
                    f"anchor_rms={d['rms_D_anchor_after']:.3e}, terminal_rms={d['rms_D_terminal']:.3e}, "
                    f"Jmin(T)={d['min_J_terminal']:.6e}, geom_hard={d['geometry_resid_rel_hard']:.3e}, "
                    f"substeps={n_sub}, dt_last={last_dt:.3e}, dt_next={dt_sub_next:.3e}, wall={d['wall_time']:.2f}s"
                )
                xs.append(x.copy()); us.append(u.copy()); ps.append(p.copy()); times.append(t_now)
                center_p.append(d['p_center']); center_pref.append(pref); center_p_fit.append(d['p_center_fit'])
                rels.append(d['rel_rms']); pmaxs.append(d['pmax']); gmaxs.append(d['gmax'])
                dp_infs.append(d['dp_inf']); dp_rels.append(d['dp_rel']); dg_infs.append(d['dg_inf']); dg_rels.append(d['dg_rel'])
                anchor_rms.append(d['rms_D_anchor_after']); terminal_rms.append(d['rms_D_terminal']); terminal_max.append(d['max_D_terminal'])
                hard_rels.append(d['hard_resid_rel']); geom_hard.append(d['geometry_resid_rel_hard']); linear_rels.append(d['linear_resid_rel'])
                min_J_star.append(d['min_J_star']); min_J_action.append(d['min_J_action']); min_J_terminal.append(d['min_J_terminal']); max_J_terminal.append(d['max_J_terminal'])
                cond_F_action.append(d['max_cond_F_action']); cond_F_terminal.append(d['max_cond_F_terminal'])
                represented_areas.append(d['represented_area']); polygon_areas.append(d['polygon_area'])
                delta_K_pressure.append(d['delta_K_pressure']); pressure_work.append(d['pressure_work_identity']); pressure_work_defect.append(d['pressure_work_identity_defect'])
                wall_times.append(d['wall_time'])
                substeps_saved.append(n_sub); dt_last_saved.append(last_dt); t_p_saved.append(last_t_p)

                # Keep --save-every as the in-memory/output-frame stride, but write
                # the rolling .npz checkpoint less often to reduce disk I/O.
                if step == 1 or step % self.args.checkpoint_every == 0:
                    write_output_checkpoint(tag=f'step_{step:05d}')
                    print('checkpoint saved:', self.args.output)

        write_output_checkpoint(tag='final')
        print('saved:', self.args.output)

def main():
    ap = argparse.ArgumentParser(
        description='Strict MLS2 comparison: D_T/G_1/2, D_star/G_star, or D_T/G_T.'
    )
    ap.add_argument('--branch', choices=['ccop-midpoint', 'star-star', 'terminal-terminal'],
                    default='ccop-midpoint')
    ap.add_argument('--backend', choices=['auto', 'cpu', 'gpu'], default='auto')
    ap.add_argument('--fp64', action='store_true', default=True,
                    help='Retained for command compatibility; strict mode always requires fp64.')
    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--resolution', type=int, default=100,
                    help='With radius=0.5, resolution=100 gives dx=0.01 and R/dx=50.')
    ap.add_argument('--init-scheme', choices=['cartesian', 'ring'], default='ring')
    ap.add_argument('--phase-jitter', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--add-bbox-points', action='store_true')
    ap.add_argument('--n-theta-outer', type=int, default=0)
    ap.add_argument('--dx', type=float, default=-1.0)
    ap.add_argument('--h-factor', type=float, default=1.3)
    ap.add_argument('--rho0', type=float, default=1.0)
    ap.add_argument('--dt', type=float, default=2.0e-3)
    ap.add_argument('--steps', type=int, default=15000,
                    help='15000 steps at dt=0.002 gives final physical time T=30.')
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--checkpoint-every', type=int, default=500,
                    help='Write rolling .npz checkpoint every N physical steps. '
                         'This is separate from --save-every, which controls how often frames are stored in memory/output arrays.')
    ap.add_argument('--delta0', type=float, default=0.4)
    ap.add_argument('--omega', type=float, default=1.2)
    ap.add_argument('--omega2', type=float, default=-1.0)
    ap.add_argument('--placement', choices=['midpoint', 'star', 'current', 'terminal'], default='midpoint')
    ap.add_argument('--theta', type=float, default=0.5)
    ap.add_argument('--position-correction', type=float, default=0.5)
    ap.add_argument('--predictor', choices=['kinematic', 'semi-implicit', 'implicit-midpoint-force'], default='implicit-midpoint-force')
    ap.add_argument('--free-surface-mode', choices=['radial', 'material', 'none'], default='material')
    ap.add_argument('--free-surface-width', type=float, default=1.0)
    ap.add_argument('--mls-reg', type=float, default=1e-8)
    ap.add_argument('--mls2-reg', type=float, default=1e-12,
                    help='Regularization for polynomial MLS derivative weights used by --reference-operator mls1/mls2.')
    ap.add_argument('--reference-operator', choices=['raw', 'corrected', 'mls1', 'mls2'], default='mls2',
                    help='Reference derivative operator for D/G/F. mls2 is second-order polynomial MLS and is recommended for quadratic pressure.')
    ap.add_argument('--operator-correction-reg', type=float, default=1e-10,
                    help='Small regularization added to the first-order corrected SPH moment matrix before inversion.')
    ap.add_argument('--geometry-iters', type=int, default=30)
    ap.add_argument('--geometry-accept-rel', type=float, default=1e-10,
                    help='Hard accepted kinematic RMS residual divided by dx.')
    ap.add_argument('--j-admissible', type=float, default=1e-6,
                    help='Raw accepted/trial-operator Jacobian must exceed this value.')
    ap.add_argument('--constant-reproduction-tol', type=float, default=1e-10)
    ap.add_argument('--lambda-step', type=float, default=1.0)
    ap.add_argument('--lambda-step-max', type=float, default=1.0)
    ap.add_argument('--lambda-step-min', type=float, default=1e-5)
    ap.add_argument('--lambda-complete-tol', type=float, default=1e-10,
                    help='Treat continuation as complete when lambda >= 1 - this tolerance.')
    ap.add_argument('--disable-stage-precheck', dest='enable_stage_precheck', action='store_false',
                    help='Disable v11 stage precheck. By default, a lambda stage is accepted without GMRES if its residual is already below the precheck tolerance.')
    ap.set_defaults(enable_stage_precheck=True)
    ap.add_argument('--stage-precheck-rel', type=float, default=1e-6,
                    help='Residual threshold for accepting an already-solved lambda stage without forcing Newton/GMRES. If <=0, use the normal stage tolerance.')
    ap.add_argument('--disable-full-lambda-snap', dest='enable_full_lambda_snap', action='store_false',
                    help='Disable v11 direct test of lambda=1. By default, if current pressure already solves the full problem, the solver snaps to lambda=1.')
    ap.set_defaults(enable_full_lambda_snap=True)
    ap.add_argument('--full-lambda-snap-rel', type=float, default=-1.0,
                    help='Residual threshold for snapping directly to lambda=1. If <=0, use --final-accept-rel.')
    ap.add_argument('--pressure-time-theta', type=float, default=-1.0,
                    help='Diagnostic pressure time level. Default infers from placement: midpoint=0.5, current=0, terminal/star=1.')
    ap.add_argument('--lambda-growth', type=float, default=1.2)
    ap.add_argument('--lambda-accept-rel', type=float, default=1e-8)
    ap.add_argument('--final-accept-rel', type=float, default=1e-9)
    ap.add_argument('--accept-rel-slack', type=float, default=0.05)
    ap.add_argument('--local-target-rel', type=float, default=2e-3)  # reserved
    ap.add_argument('--newton-iters', type=int, default=12)
    ap.add_argument('--min-newton-iters', type=int, default=1,
                    help='Minimum Newton/JVP attempts before accepting a small divergence residual. Prevents pressure under-updating.')
    ap.add_argument('--krylov', choices=['gmres', 'lgmres'], default='gmres')
    ap.add_argument('--gmres-iters', type=int, default=300)
    ap.add_argument('--gmres-restart', type=int, default=100)
    ap.add_argument('--gmres-tol', type=float, default=1e-10)
    ap.add_argument('--jvp-mode', choices=['autodiff', 'fd'], default='autodiff')
    ap.add_argument('--jvp-eps', type=float, default=1e-6)
    ap.add_argument('--newton-damp', type=float, default=0.0,
                    help='Pseudo-transient damping in matrix-free Newton matvec: Jv + newton_damp*v.')
    ap.add_argument('--pressure-update-rel-tol', type=float, default=1e-9,
                    help='Relative infinity-norm tolerance for Newton pressure update convergence.')
    ap.add_argument('--pressure-update-abs-tol', type=float, default=1e-11,
                    help='Absolute infinity-norm tolerance for Newton pressure update convergence.')
    ap.add_argument('--force-update-rel-tol', type=float, default=1e-9,
                    help='Relative infinity-norm tolerance for pressure force update convergence.')
    ap.add_argument('--force-update-abs-tol', type=float, default=1e-11,
                    help='Absolute infinity-norm tolerance for pressure force update convergence.')
    ap.add_argument('--center-fit-radius', type=float, default=0.20,
                    help='Diagnostic radius fraction of R for quadratic center-pressure fit. <=0 disables.')
    ap.add_argument('--max-delta-p-frac', type=float, default=0.25,
                    help='Trust-region bound for max |delta p| relative to max(1, max|p|).')
    ap.add_argument('--max-delta-p-abs', type=float, default=0.05,
                    help='Absolute trust-region lower bound for max |delta p|.')
    ap.add_argument('--max-beta-if-gmres-fail', type=float, default=0.25,
                    help='If GMRES/LGMRES does not converge, reject beta larger than this.')
    ap.add_argument('--beta-grid', type=str, default='1,0.75,0.5,0.25,0.125,0.0625,0')
    ap.add_argument('--min-improve', type=float, default=1e-4)
    ap.add_argument('--max-pressure-displacement-frac', type=float, default=0.2)
    ap.add_argument('--linear-accept-rel', type=float, default=1e-10,
                    help='Hard relative residual tolerance for D_star^q G_star.')
    ap.add_argument('--lu-pivot-ratio-min', type=float, default=1e-14,
                    help='Reject a star-star pressure factorization with a smaller U-pivot ratio.')
    ap.add_argument('--pressure-init', choices=['previous', 'zero'], default='previous')
    ap.add_argument('--dt-substep-init-factor', type=float, default=1.0,
                    help='Initial accepted substep size as a fraction of the nominal --dt.')
    ap.add_argument('--dt-substep-growth', type=float, default=1.25,
                    help='Growth factor for the next substep after a successful accepted substep.')
    ap.add_argument('--dt-min-factor', type=float, default=0.0625,
                    help='Minimum substep size as a fraction of nominal --dt before declaring failure.')
    ap.add_argument('--max-substeps-per-step', type=int, default=64,
                    help='Maximum accepted substeps allowed inside one nominal physical step.')
    ap.add_argument('--time-eps-factor', type=float, default=1e-10,
                    help='Relative tolerance for snapping accumulated floating-point time remainder to target step time.')
    ap.add_argument('--time-eps-abs', type=float, default=1e-14,
                    help='Absolute tolerance for snapping accumulated floating-point time remainder to target step time.')
    ap.add_argument('--allow-incomplete-step', action='store_true',
                    help='Debug only: allow advancing even if continuation does not reach lambda=1. Default is strict rejection.')
    ap.add_argument('--output', type=str, default='')
    args = ap.parse_args()

    if args.allow_incomplete_step:
        raise RuntimeError('--allow-incomplete-step is disabled in the strict comparison.')
    if args.branch == 'ccop-midpoint':
        args.placement, args.theta = 'midpoint', 0.5
    elif args.branch == 'star-star':
        args.placement, args.theta = 'star', 1.0
    elif args.branch == 'terminal-terminal':
        args.placement, args.theta = 'terminal', 1.0
    else:
        raise RuntimeError(f'Unknown branch {args.branch!r}.')
    if not args.output:
        args.output = f'mls2_{args.branch.replace("-", "_")}_Rdx50_dt0p002_T30.npz'

    expected_ratio = args.radius / (args.dx if args.dx > 0 else 2.0*args.radius/args.resolution)
    if abs(expected_ratio-50.0) > 1e-12:
        print(f'[warning] requested geometry gives R/dx={expected_ratio:.8g}, not 50.')
    if abs(args.dt*args.steps-30.0) > 1e-12:
        print(f'[warning] requested final physical time is {args.dt*args.steps:.8g}, not 30.')

    solver = PiolaPressureSolver(args)
    solver.run()


if __name__ == '__main__':
    main()
