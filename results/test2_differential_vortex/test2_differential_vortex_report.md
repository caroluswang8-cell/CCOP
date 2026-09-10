# Test 2 — analytic differential vortex under nonaffine material shear

**Decision: GO: NONAFFINE DIFFERENTIAL-VORTEX TEST 2 VALIDATED**

## Analytic reference

The pressure was re-derived from the radial momentum balance, `p(r)=rho integral_r^R s(kappa-Omega(s)^2) ds`. The map, F, J=1, Euler balance, boundary pressure, physical-vacuum sign, and shear condition-number formula were checked independently.

Analytic gate passed: `True`.
 Maximum defects: F finite-difference `2.121e-09`, J-1 `4.441e-16`, Euler balance `1.110e-16`, boundary pressure `1.110e-16`.

## 2A — prescribed-map principal evidence

| quantity | bulk rate | first-layer rate | boundary rate | global rate |
|---|---:|---:|---:|---:|
| F | 1.8577215648072063 | 1.8870830895410717 | 1.9714047868495783 | 2.0133096761554716 |
| J | 1.7822771543583555 | 1.8503987158215236 | 1.9681898800231363 | 2.2812579902548955 |
| G_physical | 1.8456748794299074 | 1.6237523599987933 | 1.8105759327262216 | 1.9989001184561708 |
| D_physical_represented | 1.830151033983784 | 1.715856513020057 | None | 1.8692906450616038 |
| D_physical_independent | 3.136102500666923 | 3.1559164589543127 | 2.605893280008979 | 3.1283121050568474 |
| DG_physical_represented | 1.634426391079921 | 0.2743967539611584 | None | 1.1528509449256192 |
| DG_physical_all_particle | 1.69351190973592 | 0.08859062396063633 | 0.7754138195840494 | 1.2661798114635978 |

2A passed: `True`. Convergence failures: `0`; stability failures: `0`.

The production divergence has only interior pressure-test rows; boundary D and DG values are therefore supplied only by the independent all-particle diagnostic and are not mislabeled as production closure.

Across all prescribed `(h, chi, theta)` cases, the minimum Riesz-scaled cross gap was `4.614e-01` and the largest condition number was `3.983e+03`; no square system lost rank. The theta=0 controls at large shear cross the sufficient positive relative-deformation margin and are reported as controls, not folded into the main theta=1/2 claim.

The coarse first-layer DG cancellation was audited rather than hidden.  The propagated-G component at the four main levels was `[0.05824665458200465, 0.055314167941624214, 0.04809080563741312, 0.04336856340633575]`; a separate h=1/36 probe confirmed continued decrease of the total independent cross-response error.

## 2B — exact-increment recursive transport

2B passed: `True`. Failures: `0`.

| layers | depth | F direct-exact | F recursive-exact | F rec-direct | ratio |
|---:|---:|---:|---:|---:|---:|
| 12 | 4 | 2.559e-02 | 2.559e-02 | 4.630e-16 | 1.809e-14 |
| 12 | 8 | 2.559e-02 | 2.559e-02 | 5.507e-16 | 2.152e-14 |
| 12 | 16 | 2.559e-02 | 2.559e-02 | 6.928e-16 | 2.707e-14 |
| 12 | 32 | 2.559e-02 | 2.559e-02 | 9.910e-16 | 3.872e-14 |
| 12 | 64 | 2.559e-02 | 2.559e-02 | 1.329e-15 | 5.192e-14 |
| 18 | 4 | 1.126e-02 | 1.126e-02 | 5.957e-16 | 5.290e-14 |
| 18 | 8 | 1.126e-02 | 1.126e-02 | 6.479e-16 | 5.754e-14 |
| 18 | 16 | 1.126e-02 | 1.126e-02 | 8.023e-16 | 7.124e-14 |
| 18 | 32 | 1.126e-02 | 1.126e-02 | 1.018e-15 | 9.038e-14 |
| 18 | 64 | 1.126e-02 | 1.126e-02 | 1.356e-15 | 1.204e-13 |
| 24 | 4 | 6.325e-03 | 6.325e-03 | 7.479e-16 | 1.182e-13 |
| 24 | 8 | 6.325e-03 | 6.325e-03 | 7.779e-16 | 1.230e-13 |
| 24 | 16 | 6.325e-03 | 6.325e-03 | 8.994e-16 | 1.422e-13 |
| 24 | 32 | 6.325e-03 | 6.325e-03 | 1.093e-15 | 1.727e-13 |
| 24 | 64 | 6.325e-03 | 6.325e-03 | 1.425e-15 | 2.253e-13 |

## 2C — full production CCOP evolution

2C passed: `True`; publication matrix complete: `True`.

| layers | dt | theta | x | u | p | F | J | phase | independent D |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.0009766 | 0.5 | 2.086e-03 | 5.278e-03 | 3.015e-03 | 2.985e-02 | 2.012e-02 | 2.356e-03 | 1.366e-02 |
| 12 | 0.0009766 | 1.0 | 2.060e-03 | 5.212e-03 | 2.987e-03 | 2.980e-02 | 2.013e-02 | 2.299e-03 | 1.366e-02 |
| 18 | 0.0009766 | 0.5 | 9.273e-04 | 2.342e-03 | 1.326e-03 | 1.368e-02 | 8.424e-03 | 1.046e-03 | 5.979e-03 |
| 18 | 0.0009766 | 1.0 | 9.001e-04 | 2.274e-03 | 1.296e-03 | 1.362e-02 | 8.429e-03 | 9.878e-04 | 5.983e-03 |
| 24 | 0.0009766 | 0.5 | 5.235e-04 | 1.321e-03 | 7.518e-04 | 7.847e-03 | 4.554e-03 | 5.902e-04 | 3.376e-03 |
| 24 | 0.0009766 | 1.0 | 4.965e-04 | 1.252e-03 | 7.213e-04 | 7.791e-03 | 4.556e-03 | 5.326e-04 | 3.378e-03 |
| 30 | 0.0009766 | 0.5 | 3.356e-04 | 8.453e-04 | 4.812e-04 | 5.108e-03 | 2.885e-03 | 3.786e-04 | 2.106e-03 |
| 30 | 0.0009766 | 1.0 | 3.091e-04 | 7.772e-04 | 4.502e-04 | 5.056e-03 | 2.887e-03 | 3.215e-04 | 2.107e-03 |
| 30 | 0.0004883 | 0.5 | 3.355e-04 | 8.453e-04 | 4.814e-04 | 5.108e-03 | 2.885e-03 | 3.786e-04 | 2.106e-03 |
| 30 | 0.0004883 | 1.0 | 3.218e-04 | 8.106e-04 | 4.659e-04 | 5.081e-03 | 2.886e-03 | 3.496e-04 | 2.107e-03 |
| 24 | 0.01562 | 0.5 | 5.248e-04 | 1.322e-03 | 7.295e-04 | 7.849e-03 | 4.554e-03 | 5.962e-04 | 3.376e-03 |
| 24 | 0.01562 | 1.0 | 5.663e-04 | 1.002e-03 | 3.296e-04 | 7.831e-03 | 4.594e-03 | 6.287e-04 | 3.410e-03 |
| 24 | 0.007812 | 0.5 | 5.238e-04 | 1.321e-03 | 7.445e-04 | 7.847e-03 | 4.554e-03 | 5.917e-04 | 3.376e-03 |
| 24 | 0.007812 | 1.0 | 4.095e-04 | 9.060e-04 | 5.119e-04 | 7.603e-03 | 4.574e-03 | 2.910e-04 | 3.393e-03 |
| 24 | 0.003906 | 0.5 | 5.236e-04 | 1.321e-03 | 7.493e-04 | 7.847e-03 | 4.554e-03 | 5.905e-04 | 3.376e-03 |
| 24 | 0.003906 | 1.0 | 4.340e-04 | 1.070e-03 | 6.292e-04 | 7.667e-03 | 4.564e-03 | 3.798e-04 | 3.384e-03 |
| 24 | 0.001953 | 0.5 | 5.235e-04 | 1.321e-03 | 7.511e-04 | 7.847e-03 | 4.554e-03 | 5.903e-04 | 3.376e-03 |
| 24 | 0.001953 | 1.0 | 4.723e-04 | 1.187e-03 | 6.903e-04 | 7.743e-03 | 4.559e-03 | 4.777e-04 | 3.380e-03 |
| 12 | 0.01562 | 0.5 | 2.088e-03 | 5.279e-03 | 2.969e-03 | 2.986e-02 | 2.012e-02 | 2.362e-03 | 1.366e-02 |
| 12 | 0.007812 | 0.5 | 2.087e-03 | 5.278e-03 | 2.996e-03 | 2.985e-02 | 2.012e-02 | 2.358e-03 | 1.366e-02 |
| 12 | 0.003906 | 0.5 | 2.086e-03 | 5.278e-03 | 3.008e-03 | 2.985e-02 | 2.012e-02 | 2.357e-03 | 1.366e-02 |
| 12 | 0.001953 | 0.5 | 2.086e-03 | 5.278e-03 | 3.013e-03 | 2.985e-02 | 2.012e-02 | 2.356e-03 | 1.366e-02 |
| 18 | 0.01562 | 0.5 | 9.286e-04 | 2.343e-03 | 1.298e-03 | 1.368e-02 | 8.424e-03 | 1.052e-03 | 5.979e-03 |
| 18 | 0.007812 | 0.5 | 9.276e-04 | 2.342e-03 | 1.316e-03 | 1.368e-02 | 8.424e-03 | 1.047e-03 | 5.979e-03 |
| 18 | 0.003906 | 0.5 | 9.274e-04 | 2.342e-03 | 1.322e-03 | 1.368e-02 | 8.424e-03 | 1.046e-03 | 5.979e-03 |
| 18 | 0.001953 | 0.5 | 9.273e-04 | 2.342e-03 | 1.325e-03 | 1.368e-02 | 8.424e-03 | 1.046e-03 | 5.979e-03 |
| 24 | 0.003906 | 0.0 | 6.519e-04 | 1.619e-03 | 8.736e-04 | 8.136e-03 | 4.544e-03 | 8.352e-04 | 3.368e-03 |

### Refinement interpretation

- theta=0.5 spatial rates: x=1.994, u=1.998, p=2.001, F=1.927, J=2.124, phase=1.995, independent_D=2.034.
- theta=0.5 temporally resolved rates: none; spatial-floor limited: x, u, p, F, J, phase, independent_D, area_J, area_polygon, energy.
- theta=0.5 fixed-cloud temporal self-difference rates: x=2.000, u=2.000, p=1.538, F=2.000, J=1.993, phase=2.001.
- theta=1 spatial rates: x=2.067, u=2.074, p=2.062, F=1.936, J=2.124, phase=2.142, independent_D=2.034.
- theta=1 temporally resolved rates: energy=1.002; spatial-floor limited: x, u, p, F, J, phase, independent_D, area_J, area_polygon.
- theta=1 fixed-cloud temporal self-difference rates: x=0.999, u=1.001, p=1.015, F=1.001, J=1.003, phase=0.997.
- Production refinement failures: `0`.

### Pressure-coefficient mixed-term audit

The midpoint coefficient differences are fitted to `||p_dt-p_dt/2|| = a_h dt + b_h dt^2`. The coefficient is audited separately from the placed vector action.

| R/h | effective rate | a_h | b_h | relative fit residual |
|---:|---:|---:|---:|---:|
| 12 | 1.234 | 1.1793e-03 | 4.4327e-02 | 1.340e-04 |
| 18 | 1.402 | 5.2165e-04 | 4.4287e-02 | 1.817e-04 |
| 24 | 1.538 | 2.9354e-04 | 4.4218e-02 | 1.643e-04 |

Observed spatial rate of `a_h`: 2.007; relative spread of `b_h`: 2.475e-03. The intermediate 1.538 rate is therefore a resolved space-time crossover, not a vector-action stage order.

## Decision reasons

- None.

## Narrow interpretation

Test 2 establishes only that the target/action configuration split, the cross-configuration response, and the transported operator realization remain accurate and stable under this strictly nonaffine material shear. It is not presented as a universal free-surface benchmark.

## Reproduction

```powershell
python test2_differential_vortex.py --phase all --production-matrix full
python -m unittest -v test_test2_differential_vortex.py
```
