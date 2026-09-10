# Manuscript-to-code map

| Manuscript evidence | Primary driver | Regression gate | Archived compact output |
|---|---|---|---|
| Case 1: affine anchor/action separation | `code/test1_affine_free_boundary.py` | `code/test_test1_affine_free_boundary.py` | `results/test1_affine_free_boundary/` |
| Case 2: long-time CCOP vs star--star ISPH | `code/card_piola_pressure_matrixfree_jax_v12_sph2_three_branch_strict.py` and MLS2 counterpart | strict in-driver acceptance checks | `fig_case2_long_time_axes_pressure_energy.pdf` |
| Case 3A--C: differential vortex | `code/test2_differential_vortex.py` | `code/test_test2_differential_vortex.py` | `results/test2_differential_vortex/` |
| Constant-field consistency | `code/constant_translation_audit.py` | direct residual and convergence checks | `results/test2_differential_vortex/constant_translation_audit.json` |

`code/ccop_time_integrators.py` contains only the conservative
implicit-midpoint predictor shared by the nonaffine driver.  It is separated
from unrelated legacy experiments so the published dependency graph remains
self-contained.

`code/check_tex_static.py` is the manuscript integrity gate used by the GitHub
workflow to verify labels, citations, environments, and figure paths.
