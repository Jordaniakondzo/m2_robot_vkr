# Modelling the motion of a nonholonomic wheeled mobile robot

Research code for a Master's research internship (Applied Informatics) and the subsequent thesis. Practice topic: «Моделирование движения колесного мобильного робота по заданным точкам траектории».

**Status.** KT4 working simulation available (built on the KT3 baseline). *Baseline v1* is the current dynamic model used for the M2 practice stage: three second-order ODEs, full state $X=(x,y,\theta,v_x,v_y,\omega)$, nonholonomic constraint $r(X)=v_x\sin\theta-v_y\cos\theta=0$, numerical integration, planar simulation and six-component target states $T_i=X(t_i)$ (the supervisor's minimum for the practice: theory plus simulation of a wheeled robot on a plane). It may be further refined, particularly regarding numerical preservation of the nonholonomic constraint (an Erugin–Galiullin-type construction is to be studied; not yet validated). It is not a wheel-drive or Ackermann model. All parameters are synthetic and all results are simulation outputs, not measurements of a physical robot. Neural control (e.g. $\dot\phi$ as output) belongs to the later thesis stage and is out of the current scope. The simulation integrates the model under constant inputs $F,\tau$; no feedback or target-reaching control law is implemented, so reaching prescribed target states remains an open later problem.

## Model

Planar rigid body with an ideal no-lateral-slip constraint. State $X=(x,y,\theta,\dot x,\dot y,\dot\theta)\in\mathbb R^6$, inputs $F$ [N] (longitudinal force) and $\tau$ [N m] (yaw torque), parameters $m$ [kg], $J$ [kg m²]. With $v=\dot x\cos\theta+\dot y\sin\theta$ and $\omega=\dot\theta$:

$$m\ddot x=F\cos\theta-m v\omega\sin\theta,\quad m\ddot y=F\sin\theta+m v\omega\cos\theta,\quad J\ddot\theta=\tau,$$

with the constraint residual $r(X)=\dot x\sin\theta-\dot y\cos\theta=0$. The residual is a first integral of the model: numerically, its drift $|r(t)-r(0)|$ measures integration error only. A 5-state form $(x,y,\theta,v,\omega)$ is used solely as a cross-check. Derivation and checks: `notebooks/01_model_and_validation.ipynb`.

## Layout

```text
src/model.py        equations of motion, 5-state cross-check, closed-form free motion
src/solver.py       checked wrapper of scipy solve_ivp (method, rtol, atol required)
src/metrics.py      constraint residual and drift, trajectory error, energy balance
src/dataset.py      target states T_i = X(t_i): sampling, CSV/JSON export and reload
src/plots.py        paths, speeds, residual drift; robot pose geometry, planar-motion figure, animation
experiments/scenarios.py  scenario definitions, parameters, solver settings and runner shared by KT3 and KT4
experiments/run_kt3.py   the three KT3 scenarios -> targets, diagnostics, figures
experiments/run_kt4.py   KT4 working simulation -> planar-motion figures, diagnostics, optional animation
notebooks/01_model_and_validation.ipynb   derivation, symbolic check, validation
tests/              model identities, independent validation, dataset round trip, visualization checks
results/            reference runs: kt3_<UTC>/ and kt4_<UTC>/ (timestamped; never overwritten)
```

## Scenarios (KT3, reused in KT4)

| id | $F,\tau$ | reference |
|---|---|---|
| S1_line | 0, 0 | closed form (straight line) |
| S2_arc | 0, 0 | closed form (circle) |
| S3_const_F_tau | 2 N, 0.2 N m | no closed form for $x(t),y(t)$ (see below); 5-state cross-check |

$m=10$ kg, $J=2$ kg m²; DOP853, `rtol=1e-10`, `atol=1e-12`; targets at $t=2,4,6,8,10$ s.

For S3, no closed-form solution is used for the coordinates $x(t),y(t)$. The trajectory is verified independently from the analytic expressions of $v(t)=v_0+Ft/m$, $\omega(t)=\omega_0+\tau t/J$ and $\theta(t)=\theta_0+\omega_0 t+\tau t^2/(2J)$, then by numerical quadrature of $x(t)$ and $y(t)$ (`tests/test_validation.py`, notebook section 8).

## Usage

Python ≥ 3.11.

```bash
pip install -e ".[dev]"
pytest
python -m experiments.run_kt3          # writes results/kt3_<UTC>/
python -m experiments.run_kt4          # writes results/kt4_<UTC>/
python -m experiments.run_kt4 --save-animation   # additionally writes S2_arc_animation.gif
pip install -e ".[notebook]"           # to run the notebook
```

Each run records parameters, initial state, solver and tolerances, git commit and diagnostics in `metadata.json`.

## KT4 outputs

`python -m experiments.run_kt4` runs the three scenarios above and writes a new `results/kt4_<UTC>/` directory (reference run: `results/kt4_20261110T112622Z/`):

- `S1_line_motion.png`, `S2_arc_motion.png`, `S3_const_F_tau_motion.png`: planar motion $(x(t),y(t))$ with the robot drawn at five instants using the orientation $\theta(t)$. The robot rectangle (0.40 × 0.24 m) is a graphical marker of the pose $(x,y,\theta)$, not a physical or wheel model.
- The circles on these figures are the projections $(x_i,y_i)$ of the states $T_i=X(t_i)$. They are sampled from the simulated trajectory itself and are not a demonstration of a target-reaching controller.
- `speeds.png`, `residuals.png`: $v(t)$, $\omega(t)$ and the constraint drift $|r(t)-r(0)|$.
- `diagnostics.json`: per-scenario diagnostics (`max_abs_r_dense`, `max_drift_dense`, `max_state_diff_vs_5state_model`, `max_state_error_vs_closed_form` where a closed form exists), reported without pass/fail thresholds.
- `targets.csv`, `metadata.json`: target states and run metadata (parameters, solver, git commit, command).
- `S2_arc_animation.gif` (only with `--save-animation`): animation of the S2_arc pose along the simulated trajectory.

All outputs are synthetic simulation results.

## Test tolerances

The numerical thresholds in `tests/` (state errors `1e-8`, residual and energy-balance errors `1e-9`/`1e-8`) are technical software-validation thresholds. They are chosen well above the errors observed with DOP853 at `rtol=1e-12`, `atol=1e-14` (at most about `4e-11` for states and `2e-10` for the energy balance, which includes the Simpson quadrature error), i.e. with a margin of at least a factor 40. They are **not** physical tolerances and have not been validated as such by the supervisor. No tolerance-calibration policy is attached to them.

## History

The public repository starts from the consolidated KT3 baseline.

The reference dataset `results/kt3_20261027T073745Z/` retains the historical label "Baseline v1 (provisional)" because it records the exact metadata at generation time. The scientific status of Baseline v1 was later clarified as the current M2 practice model, without changing the equations, scenarios or reference results.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
