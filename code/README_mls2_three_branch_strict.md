# Strict MLS2 configuration-assignment comparison

The driver `card_piola_pressure_matrixfree_jax_v12_mls2_three_branch_strict.py`
preserves the legacy quadratic MLS2 discretization and adds only two matched
controls around the reference CCOP midpoint branch:

| branch | pressure-test target | pressure action |
|---|---:|---:|
| `ccop-midpoint` | $D_T^q$ | $G_{1/2}$ |
| `star-star` | $D_\star^q$ | $G_\star$ |
| `terminal-terminal` | $D_T^q$ | $G_T$ |

`star-star` is the matched ISPH-like predictor-configuration control.  The
legacy `--placement star` was not this control because it still solved a
terminal-divergence residual.

## Locked default case

- radius `0.5`;
- resolution `100`, hence `dx=0.01` and `R/dx=50`;
- `dt=0.002`;
- `steps=15000`, hence final physical time `T=30`;
- ring cloud, material free-surface pressure nodes, quadratic MLS2;
- mandatory fp64;
- no pressure-matrix shift, pseudo-inverse, or accepted-state determinant clipping.

## PyCharm terminal commands

From the repository root, use the Python interpreter that contains JAX:

```powershell
$py = '.\.venv\Scripts\python.exe'
$code = '.\code\card_piola_pressure_matrixfree_jax_v12_mls2_three_branch_strict.py'
```

Reference CCOP midpoint branch:

```powershell
& $py -u $code --backend cpu --branch ccop-midpoint --output mls2_ccop_midpoint_Rdx50_dt0p002_T30.npz
```

Matched predictor-configuration control:

```powershell
& $py -u $code --backend cpu --branch star-star --output mls2_star_star_Rdx50_dt0p002_T30.npz
```

Terminal-action control:

```powershell
& $py -u $code --backend cpu --branch terminal-terminal --output mls2_terminal_terminal_Rdx50_dt0p002_T30.npz
```

The defaults already impose the requested resolution, time step, and final
time.  Explicit versions of the same options are `--resolution 100 --dt 0.002
--steps 15000`.

## Acceptance and diagnostics

The nonlinear branches use exact JAX JVP, a frozen $D_\star^qG_\star$
preconditioner that does not modify the residual, a hard geometry rebuild, and
raw accepted-state Jacobian checks.  The frozen `star-star` system uses sparse
LU and rejects singular or pivot-defective systems without a shift.

The NPZ history distinguishes:

- `rms_D_anchor_after`: closure of the branch's own pressure-test target;
- `rms_D_terminal`: the common rebuilt terminal diagnostic;
- `hard_resid_rel` and `geometry_resid_rel_hard`;
- `min_J_star`, `min_J_action`, `min_J_terminal`, `max_J_terminal`;
- represented and polygonal area;
- pressure-induced kinetic-energy change and its exact algebraic work identity;
- pressure solve and wall-time diagnostics.

The code rejects failed nonlinear solves rather than silently using a smaller,
branch-dependent physical time step.
