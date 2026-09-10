# Figure contract and QA notes

## Case 2

- Core conclusion: terminal--terminal closure alone leaves the long-time pressure-work trajectory close to the star--star workflow, whereas midpoint pressure action restores the affine reference and refines systematically in time.
- Results-level question: which configuration assignment controls the long-time state, center pressure, and conservative energy exchange when the spatial fields are polynomially reproduced?
- Archetype: quantitative comparison grid.
- Hero evidence: the common terminal-divergence panel is read together with the mechanical-energy and semi-axis panels.
- Controls: star--star and terminal--terminal use the same spatial discretization and time step; three midpoint time steps expose temporal refinement.
- Statistics: deterministic trajectories; no stochastic averaging or uncertainty intervals.
- Exclusion: only the archived t=0 pressure/divergence placeholders are omitted because they precede the first pressure solve.

### Panel audit

| Panel | Unique evidence role | Quantity | Variability | Collision QA | Visual QA |
|---|---|---|---|---|---|
| a | Long-time deformation | major/minor semi-axes | deterministic, none | pass | pass |
| b | Pressure response | center-pressure error | deterministic, none | pass | pass |
| c | Energy component | kinetic error | deterministic, none | pass | pass |
| d | Energy component | potential error | deterministic, none | pass | pass |
| e | Work consequence | mechanical-energy drift | deterministic, none | pass | pass |
| f | Constraint target | terminal-divergence RMS | deterministic, none | pass | pass |

## Case 3

- Core conclusion: under strictly nonaffine shear, both placements retain algebraic terminal closure while the physical state, pressure, deformation phase, and independent divergence approach the analytic solution under spatial refinement.
- Results-level question: does the configuration-separated method retain its physical and algebraic behavior when the material map varies in space?
- Archetype: quantitative validation grid.
- Hero evidence: radial phase and final spatial convergence; energy-component and divergence histories provide orthogonal validation.
- Statistics: deterministic trajectories; no stochastic averaging or uncertainty intervals.
- The exact map preserves the circular boundary, so major/minor axes are identically non-discriminating and radial phase is used instead.
- The archived result stores global pressure error, not a center-pressure time series; no unrecorded center value is reconstructed.
- The archived energy diagnostics are absolute relative invariant errors; no sign is inferred.

### Panel audit

| Panel | Unique evidence role | Quantity | Variability | Collision QA | Visual QA |
|---|---|---|---|---|---|
| a | Nonaffine deformation observable | radial angular displacement | deterministic, none | pass | pass |
| b | Spatial refinement | final RMS errors | deterministic, none | pass | pass |
| c | Kinetic invariant | relative error | deterministic, none | pass | pass |
| d | Potential invariant | relative error | deterministic, none | pass | pass |
| e | Mechanical invariant | relative error | deterministic, none | pass | pass |
| f | Algebraic/physical divergence separation | represented and independent RMS | deterministic, none | pass | pass |

## Data integrity

- All saved observations in the selected trajectories are plotted or exported to source-data CSV files.
- No downsampling, smoothing, clipping, pressure alignment, or synthetic uncertainty is applied.
- The analytic affine curves are evaluated by an independent DOP853 solve at the archived output times.
- Input paths are not written into public outputs; only archive basenames are retained.

## Automated and visual QA

- Source preflight: 20 passes, no warnings or failures.
- PDF glyph audit: minimum rendered text size 5.18 pt in both figures; pass.
- Rendered collision audit: zero failures and zero warnings in both figures; pass.
- Each panel was inspected at final double-column size for legibility, hierarchy, curve identity, and clipping; pass.

## Machine-readable summary pointers

- Case 2 records: 1660.
- Case 3 spatial runs: 8.
