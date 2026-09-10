# Test 1 — force-free affine incompressible liquid in vacuum

**Decision: GO: AFFINE FREE-BOUNDARY TEST 1 VALIDATED**

## Literature and exact reference

The zero-magnetic-field (`kappa=0`) equations were checked directly against Roberts–Shkoller–Sideris. The paper gives `A_ddot=-kappa A+lambda cof(A)` and `lambda=2(kappa-det(A_dot))/|A|_F^2`; hence the requested perfect-fluid signs and pressure normalization are consistent.

The scalar DOP853 reference has max `|det A-1|` 1.110e-16 and max matrix-energy error 6.661e-16.

## Pressure space and operator audit

| h target | N | boundary | pressure dofs | relative gap | condition | max GMLS condition |
|---:|---:|---:|---:|---:|---:|---:|
| 1/12 | 499 | 76 | 423 | 3.848e-03 | 2.599e+02 | 3.965e+01 |
| 1/18 | 1087 | 114 | 973 | 2.544e-03 | 3.931e+02 | 3.938e+01 |
| 1/24 | 1901 | 152 | 1749 | 1.816e-03 | 5.508e+02 | 3.937e+01 |
| 1/30 | 2939 | 188 | 2751 | 1.229e-03 | 8.139e+02 | 3.910e+01 |

Boundary pressure values are fixed to zero and are not unknowns. The square production system uses the matching interior divergence rows. No pin, pseudo-inverse, shift, clipping, filtering, or hidden projection is used.

## Test 1A — terminal anchors

| anchor | terminal residual rate | source-mismatch rate | J-1 rate |
|---|---:|---:|---:|
| n | 1.000725519532569 | 0.0007255195325687804 | 1.9999999999991398 |
| star | 2.9991549558998503 | 1.9991549558998492 | 3.9990950155305653 |
| T | 0.37061810651383326 | -0.6293818934861684 | 3.998372705127107 |

Maximum `D_T`-anchored represented residual: 4.364e-15.

## Test 1B — pressure placement

| h target | dK(0) | dK(1/2) | dK(1) | theta* | max closure | work defect |
|---:|---:|---:|---:|---:|---:|---:|
| 1/12 | 3.151e-04 | -6.296e-08 | -3.149e-04 | 0.499900069927 | 9.558e-16 | 2.285e-16 |
| 1/18 | 3.145e-04 | -6.283e-08 | -3.143e-04 | 0.499900069927 | 1.490e-15 | 1.184e-16 |
| 1/24 | 3.143e-04 | -6.279e-08 | -3.141e-04 | 0.499900069927 | 2.060e-15 | 2.122e-16 |
| 1/30 | 3.142e-04 | -6.277e-08 | -3.140e-04 | 0.499900069927 | 2.848e-15 | 1.327e-16 |

Measured temporal rate of `|theta*-1/2|`: 1.9987346420063332.

## Test 1C — frozen projector

Endpoint `||Pi_T||_M` = 1.61863; endpoint Green/adjoint relative defect = 2.484e-01. The endpoint norm is therefore reported as a discretization baseline; only `||Pi_theta-Pi_T||_M` and the action-space angle are attributed to placement.
In the auxiliary mass-adjoint endpoint control, `||Pi_T^ad||_M` = 1, while the largest off-endpoint norm is 1.00019993967. This control is diagnostic only and leaves the production pair unchanged.

## Test 1D — exact-start placed-action stage consistency

| theta | action-stage rate | instantaneous-p rate | average-p rate | action-path rate |
|---:|---:|---:|---:|---:|
| 0 | 1.001 | 1.001 | 1.022 | exact |
| 0.5 | 1.967 | 1.969 | 1.986 | 1.992 |
| 1 | 1.010 | 1.007 | 1.031 | 2.994 |

The exact pressure applied through the numerical production-path action has maximum RMS reproduction error `8.450e-16`. The stage test starts from the exact state at nonzero time, so its rates are local rather than accumulated trajectory errors.
The endpoint coefficient errors remain first order even when measured against the exact step-average pressure.  Hence the multiplier is not treated as a configuration-independent step-average pressure; the order-bearing quantity is the placed vector action `G_theta p_theta`.

## Test 1E — affine-reference evolution

| layers | dt | theta | x error | u error | p(end) | p(action) | F error | J error | area error | energy error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 0.025 | 0.5 | 2.301e-05 | 1.267e-05 | 4.227e-03 | 1.250e-05 | 4.599e-05 | 4.983e-05 | -4.983e-05 | -2.803e-05 |
| 18 | 0.025 | 1.0 | 9.204e-04 | 3.096e-03 | 2.390e-03 | 2.390e-03 | 1.839e-03 | 5.058e-05 | -5.058e-05 | -6.775e-03 |
| 18 | 0.0125 | 0.5 | 5.753e-06 | 3.167e-06 | 2.098e-03 | 3.186e-06 | 1.150e-05 | 1.246e-05 | -1.246e-05 | -7.007e-06 |
| 18 | 0.0125 | 1.0 | 4.575e-04 | 1.552e-03 | 1.186e-03 | 1.186e-03 | 9.143e-04 | 1.255e-05 | -1.255e-05 | -3.400e-03 |
| 18 | 0.00625 | 0.5 | 1.438e-06 | 7.916e-07 | 1.045e-03 | 8.037e-07 | 2.874e-06 | 3.114e-06 | -3.114e-06 | -1.752e-06 |
| 18 | 0.00625 | 1.0 | 2.281e-04 | 7.772e-04 | 5.909e-04 | 5.909e-04 | 4.558e-04 | 3.126e-06 | -3.126e-06 | -1.703e-03 |
| 18 | 0.003125 | 0.5 | 3.595e-07 | 1.979e-07 | 5.214e-04 | 2.018e-07 | 7.185e-07 | 7.786e-07 | -7.786e-07 | -4.379e-07 |
| 18 | 0.003125 | 1.0 | 1.139e-04 | 3.888e-04 | 2.949e-04 | 2.949e-04 | 2.275e-04 | 7.801e-07 | -7.801e-07 | -8.525e-04 |
| 12 | 0.00625 | 0.5 | 1.440e-06 | 7.924e-07 | 1.060e-03 | 8.157e-07 | 2.874e-06 | 3.114e-06 | -3.114e-06 | -1.755e-06 |
| 12 | 0.00625 | 1.0 | 2.283e-04 | 7.779e-04 | 5.997e-04 | 5.997e-04 | 4.558e-04 | 3.126e-06 | -3.126e-06 | -1.707e-03 |
| 24 | 0.00625 | 0.5 | 1.438e-06 | 7.913e-07 | 1.037e-03 | 7.978e-07 | 2.874e-06 | 3.114e-06 | -3.114e-06 | -1.750e-06 |
| 24 | 0.00625 | 1.0 | 2.280e-04 | 7.769e-04 | 5.866e-04 | 5.866e-04 | 4.558e-04 | 3.126e-06 | -3.126e-06 | -1.702e-03 |
| 30 | 0.00625 | 0.5 | 1.437e-06 | 7.912e-07 | 1.033e-03 | 7.944e-07 | 2.874e-06 | 3.114e-06 | -3.114e-06 | -1.750e-06 |
| 30 | 0.00625 | 1.0 | 2.280e-04 | 7.768e-04 | 5.840e-04 | 5.840e-04 | 4.558e-04 | 3.126e-06 | -3.126e-06 | -1.702e-03 |

Temporal and spatial fitted rates are stored verbatim in the JSON. Bulk, first-layer, boundary, and all-particle independent divergence histories are also retained for every multistep run.

The independent affine-divergence values are at the fp64 differentiation floor (the finest boundary value is O(1e-13), while the explicit condition-number/h roundoff envelope is O(1e-11)). They therefore do not define a spatial convergence rate in this exactly reproduced affine test. The absence of a measurable spatial truncation regime is reported, not promoted as high-order convergence.

## Gate failures

- None.

## Scope

This result supports only terminal closure and pressure-action placement for the smooth affine free-boundary solution. It is not a non-affine spatial validation and the motion is not described as periodic or oscillatory.
