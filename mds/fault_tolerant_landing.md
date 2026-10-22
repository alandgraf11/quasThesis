# Fault-Tolerant Landing Check – Decision Template

**Question the check answers:** Starting from a state `s0`, can the vehicle still land safely inside the landing zone despite an actuator fault?

**Usage:** Pick one option per decision point (combinations are allowed for **5**, e.g. `5a+5c`). Section 3 has sub-points; a preset is enough there, optionally with overrides (e.g. `3: R, but 3.5b`). Then all that is needed is:

> Implement `mds/fault_tolerant_landing.md` with: 1a, 2a, 3: R, 4a, 5a+5c, 6c, 7a, 8a, 9b, 10b
> (+ numerical values from the section "Open numerical values", if needed)

---

## 0. Fixed part (always implemented)

This part does not depend on any decision:

- **New files, existing code stays unchanged:** `src/ocp_landing.py` (class `OCP_landing`, modelled on `OCP_tracking`) and `main_landing.py` (single case). `main_sim.py`, `main_ocp.py`, `main_multiprocessing.py` and `OCP_tracking` are not touched.
- **New config block `landing:`** with the landing zone center `center: [x_c, y_c]`, the **landing radius `radius`**, the tolerances and the fault cases.
- **Landing zone = circle with radius `r_land`** around `(x_c, y_c)` on the surface:
  `(x_N − x_c)² + (y_N − y_c)² ≤ r_land²` as an inequality constraint in `g`.
  Center and radius are **NLP parameters `p`** (like `s0`). They can therefore be changed per case without rebuilding the solver, and can be swept later.
- **Ground:** surface at `z = 0`. For all nodes `z_k ≥ 0` (in `lbx`) and at the end `z_N = 0`. With 1d instead `‖r_k‖ ≥ R_moon` and `‖r_N‖ = R_moon`.
- **Faults are modelled via bounds:** A fault is a dict applied to copies of `lbx`/`ubx` (e.g. `Fz_max *= 0.2`, `τx_low = τx_high = 0`). Since `lbx`/`ubx` are only passed in `solve()`, **no solver rebuild is needed**.
- **Consistent initial state:** If the force and torque states in `s0` lie outside the new bounds (e.g. `Fz = 15 N`, new limit 10 N), they are clipped to the new limits. This corresponds to an instantaneous loss of performance; otherwise the OCP would already be infeasible because of `s0` (cf. note in the README).
- **"Landed" is not the same as "IPOPT successful":** `check_landing(...)` only returns `landed = True` if IPOPT converged **and** the final state `S_opt[-1]` passes the post-check. All quantities that are fixed or bounded in section 3 are checked; fixed values with a small numerical tolerance. Return dict: `landed`, `solver_success`, `status`, distance to the zone center, and for every quantity from section 3 its final value and whether it satisfies its condition, plus cost and computation time.
- **Which final-state quantities are fixed, bounded or free** is decided entirely in section 3.
- **Plots/animation:** The existing `plot_results` / `visualize_3d` are reused; the landing zone is drawn as a circle.

---

## 1. World / geometry

**1a – Flat 3D (status quo + ground)**
The existing dynamics stay unchanged; only the ground and the landing zone are added.
- Pro: minimal effort, all existing tools (plots, animation, batch) work right away.
- Con: no curvature. For landings from ≲ a few km altitude and range this is negligible though (R_moon ≈ 1737 km).

**1b – Planar 2D as a restriction of the 3D model**
Same 19-state model, but `y, v, φ, ψ, p, r, Fy, τx, τz` are fixed to 0 via bounds or constraints. Motion only happens in the x-z plane; the landing zone is an interval `|x_N − x_c| ≤ r_land`.
- Pro: fewer degrees of freedom, easier to interpret, code stays compatible.
- Con: the solver still computes with 19 states, so it is hardly faster. Faults in y or about x/z cannot be represented.

**1c – Separate reduced 2D model**
New class `Vehicle2D` with state `[x, z, vx, vz, θ, ω, Fx, Fz, τ]` (9) and control `[dFx, dFz, dτ]` (3). Angle used directly instead of a quaternion, landing zone as an interval.
- Pro: fast, very clear, well suited for large sweeps and thesis plots.
- Con: a second model that has to be maintained; plots and animation must be adapted.

**1d – 3D with curvature (spherical moon)**
Position in the moon-fixed frame relative to the moon's center. The surface is the sphere `‖r‖ = R_moon`, the landing zone a spherical cap (angular distance to the zone center ≤ `r_land / R_moon`). "Upright" means: body z points in the local radial direction. Moon rotation (Coriolis) is optional but neglected.
- Pro: physically correct for large ranges, e.g. descent from orbit.
- Con: considerably more effort (frames, initial states, plots). Numerically delicate because `R_moon` ≫ flight distance; scaling or a local tangent frame with curvature correction would be needed.

---

## 2. Gravity model

**2a – Constant:** `g = 1.62 m/s²` in −z (status quo). With 1d: constant magnitude towards the moon's center.

**2b – Altitude-dependent, fixed direction:** `g(z) = μ / (R_moon + z)²` in −z (μ_moon ≈ 4.9048·10¹² m³/s²). Fits 1a–1c. The difference to 2a only reaches the per-mille range at km altitudes.

**2c – Central field:** `g(r) = −μ · r / ‖r‖³`; direction and magnitude depend on position. **Only meaningful with 1d.**

Implemented in each case in `Vehicle.ca_model()` (the line with `wdot`). For this, `g` moves from the `Vehicle` default into the config (`vehicle.g`).

---

## 3. Landing state – what is fixed, what stays free?

The final state `s_N` has 19 components. For each one it is decided individually:
- **fixed:** exact value (equality)
- **range:** tolerance (inequality)
- **free:** no condition

The horizontal position is already defined by the landing zone (section 0). Everything else is decided here.

**Principle:** A quantity is only fixed if there is a **physical or mission-related reason** for it. Every additional condition makes landings "impossible" that would be physically safe. The results then mix the effect of the fault with the strictness of the requirement (example: 3.5).

**Selection:** either a preset (end of this section) such as `3: R`, a preset with overrides such as `3: R, but 3.5b`, or each sub-point individually.

### Overview

| Component | Index in `s` | Sub-point |
|---|---|---|
| Enforcement (hard/soft) | – | 3.0 |
| x, y | 0–1 | Section 0 (landing zone) |
| z | 2 | 3.1 |
| w (vertical) | 5 | 3.2 |
| u, v (horizontal) | 3–4 | 3.3 |
| Tilt | 6–9 (from q) | 3.4 |
| Yaw angle ψ | 6–9 (from q) | 3.5 |
| p, q (tilt rates) | 10–11 | 3.6 |
| r (yaw rate) | 12 | 3.7 |
| Fx, Fy, Fz | 13–15 | 3.8 |
| τx, τy, τz | 16–18 | 3.9 |
| Behaviour after touchdown | – | 3.10 |

**On attitude:** The quaternion is not fixed component-wise but decomposed into two quantities:
- **Tilt θ:** tilt angle of the body z-axis against the vertical
- **Yaw angle ψ:** rotation about the vertical

Reason: `q` and `−q` describe the same attitude. A condition such as `qw_N = 1` would exclude half of the valid solutions and can make the solver fail unnecessarily. All attitude conditions are therefore formulated via the rotation matrix `R(q)`, which is identical for `q` and `−q`.

### 3.0 Enforcement

**3.0a – Hard:** All chosen conditions are constraints in the OCP; landable means the OCP is feasible.
- Pro: clean feasibility statement. Con: If the OCP is infeasible, you cannot see how close it was.

**3.0b – Soft:** Only the ground and the landing zone are hard; everything else is a penalty term in the cost. Whether it landed is decided by the post-check (section 0).
- Pro: robust, and you can see the distance to landability. Con: The penalty weights influence the result, and feasibility only lives in the post-check.

**3.0c – Mixed:** Safety-critical quantities (3.1, 3.2, 3.4) are hard, all others soft.

### 3.1 Touchdown altitude `z_N`

**3.1a – `z_N = 0`:** Touchdown exactly at the last node.

**3.1b – `0 ≤ z_N ≤ h_cut`:** The engine is cut off shortly above the ground (Apollo: contact probes ≈ 1.7 m); the rest is free fall. The actual touchdown velocity is computed afterwards (`w_touch = −√(w_N² + 2·g·z_N)`, drift accordingly), and the conditions from 3.2/3.3 apply to these values.
- Pro: more realistic ending. Con: somewhat more effort; only fits together with 3.8b (engine off).

### 3.2 Vertical velocity `w_N`

**3.2a – `w_N = 0`:** Touchdown without sink rate. Ideal, but not necessary in reality, since the landing legs damp the impact.

**3.2b – `−v_z,max ≤ w_N ≤ 0`:** Residual sink rate up to the load capacity of the legs (order of magnitude Apollo LM: up to ≈ 3 m/s).

"Free" is not a sensible option here.

### 3.3 Horizontal velocity `(u_N, v_N)`

Physical reason for a condition: residual horizontal velocity can make a leg "catch" on the ground so that the vehicle tips over (order of magnitude Apollo LM: ≈ 1.2 m/s).

**3.3a – `u_N = v_N = 0`**

**3.3b – `‖(u_N, v_N)‖ ≤ v_h,max`:** Circular condition, independent of direction.

**3.3c – `|u_N|, |v_N| ≤ v_h,max`:** Box via `lbx`/`ubx`. Linear and therefore numerically simpler, but up to √2 more is allowed diagonally.

### 3.4 Tilt `θ_N`

`cos θ = R₃₃(q) = 1 − 2(qx² + qy²)`. This is independent of the yaw angle and of the sign of `q`.
Physical reason: tip-over stability on the legs (depends on leg spread and center-of-mass height) or alignment with the terrain slope.

**3.4a – `θ_N = 0`:** `qx_N = qy_N = 0`.

**3.4b – `θ_N ≤ θ_max`:** `qx_N² + qy_N² ≤ sin²(θ_max/2)`.

### 3.5 Yaw angle `ψ_N`

This is where the trade-off matters most:
- **For "free":** For a rotationally symmetric lander or a symmetric leg arrangement, ψ affects neither tip-over stability nor leg loads on flat terrain.
  **Consequence for the results:** If ψ is fixed, a τz failure (5c) automatically counts as "not landable" as soon as `ψ_0 ≠ ψ_target`, even though the vehicle could physically land without problems. The result then measures the requirement, not the fault tolerance.
- **For "fixed":** Mission requirements, e.g. leg alignment with the slope, solar panel towards the sun, antenna towards Earth, camera, payload or ramp in a specific direction, or an asymmetric leg arrangement.

**3.5a – Free**

**3.5b – Fixed `ψ_N = ψ_t`:** The horizontal projection of the body x-axis `(R₁₁, R₂₁)` must point in direction `ψ_t`:
`R₁₁·sin ψ_t − R₂₁·cos ψ_t = 0` and `R₁₁·cos ψ_t + R₂₁·sin ψ_t ≥ 0`.

**3.5c – Range `|ψ_N − ψ_t| ≤ Δψ`:** `R₁₁·cos ψ_t + R₂₁·sin ψ_t ≥ cos(Δψ)·√(R₁₁² + R₂₁²)`.

**3.5d – Symmetry range:** With n identical legs, ψ only matters modulo 360°/n (4 legs: ψ_t, ψ_t + 90°, …). The condition is `cos(n·(ψ_N − ψ_t)) ≥ cos(n·Δψ)`, also formulated via `R₁₁, R₂₁`. This is a compromise between 3.5a and 3.5c and requires considerably less yaw motion.

### 3.6 Tilt rates `(p_N, q_N)`

Physical reason: residual rotation about the tilt axes at touchdown promotes tipping over.

**3.6a – `p_N = q_N = 0`**

**3.6b – `‖(p_N, q_N)‖ ≤ ω_max`**

### 3.7 Yaw rate `r_N`

Physical reason: rotation on the legs creates lateral forces and friction at the footpads. This is considerably less critical than tilt rates.

**3.7a – `r_N = 0`**

**3.7b – `|r_N| ≤ r_max`**

**3.7c – Free:** only sensible if small residual rates are considered harmless.

### 3.8 Engine forces at touchdown `(Fx, Fy, Fz)_N`

The model has no ground contact: the last node is the touchdown instant; what happens afterwards is not simulated. The forces in the final state do, however, determine what would happen in the next moment. If the vertical thrust exceeds the weight, the vehicle lifts off again; lateral forces push it across the ground.

**3.8a – Free:** simple, but the final state can be physically inconsistent (e.g. full thrust at `z = 0`, `v = 0`).

**3.8b – Engine off `F_N = 0`:** realistic (cutoff at ground contact). Because of the rate limits (`dFz` = 20 N/s) the thrust must be ramped down beforehand, which shapes the final flight phase. Fits naturally with 3.1b.

**3.8c – Able to rest:** `(R·F_N)_z ≤ m·g` and horizontal world force `(R·F_N)_{x,y} = 0`. The vehicle stays put whether or not the engine is still running; this is the minimal physically consistent condition.

**3.8d – Hover:** `R·F_N = (0, 0, m·g)`. The final state is a hover equilibrium right at the ground. Strict.

### 3.9 Torques at touchdown `τ_N`

**3.9a – Free**

**3.9b – `τ_N = 0`:** consistent with small final rates. Otherwise the vehicle starts rotating at the moment of touchdown.

**3.9c – `|τ_i,N| ≤ τ_rest`**

### 3.10 After touchdown

**3.10a – End at node N:** Only the final state counts (default).

**3.10b – Hold phase:** The last M nodes must hold the landing state (`z = 0`, `v = 0`, `ω = 0`). This automatically enforces consistent forces and torques and makes 3.8/3.9 ineffective.
- Con: Without a ground contact model the vehicle has to actively hover at the ground for this. Physically this is a workaround and consumes thrust; with a severe thrust loss (5a) it distorts the result.

(A real ground contact model with normal force and friction would be a separate extension. It is non-smooth and unsuitable for IPOPT, so it is not an option here.)

### Presets for section 3

| Preset | 3.0 | 3.1 | 3.2 | 3.3 | 3.4 | 3.5 | 3.6 | 3.7 | 3.8 | 3.9 | 3.10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **S** – strict | a | a | a | a | a | b | a | a | d | b | a |
| **R** – realistic | a | a | b | b | b | a | b | b | c | b | a |
| **M** – minimal | b | a | b | b | b | a | b | c | a | a | a |

- **S** corresponds to "everything to zero", including a fixed yaw direction. Good as a reference, but the τz fault is then almost always "not landable" (see 3.5).
- **R** only fixes what is physically justified and keeps the final state consistent (able to rest, no residual torques).
- **M** is the most robust preset (soft), for getting first results.

---

## 4. Nominal actuator set

Important for how meaningful the fault cases are: the model currently has **independent lateral thrust** `Fx, Fy ∈ [−3, 3] N`. This lets the vehicle manoeuvre sideways without any rotation, and the failure of a torque axis is then mostly harmless.

**4a – As before:** main engine `Fz ∈ [0, 20]`, lateral thrust `Fx, Fy ∈ [−3, 3]`, torques `τ ∈ [−1, 1]`.

**4b – Main engine + torques only (realistic lander):** `Fx = Fy = 0` fixed. Horizontal motion is only possible by tilting the thrust vector, so torque faults directly affect landability.

**4c – Both:** The fault cases are computed for 4a and 4b, as a comparison in the thesis.

---

## 5. Fault cases (combinations allowed, e.g. `5a+5c`)

Faults are defined as a list in the config and are implemented only via bounds (section 0).

**5a – Reduced main engine:** `Fz_max ← k · Fz_max` (residual thrust factor k, see 6).
**5b – Lateral thrust failure:** one axis (`Fx` *or* `Fy` = 0) or both. Only relevant with 4a/4c.
**5c – Torque axis failure:** `τ_i ≡ 0` for one axis i ∈ {x, y, z}, each axis as its own case.
**5d – Reduced torques:** `|τ_i| ≤ k · τ_max` (one or all axes).
**5e – Slower actuators:** rate limits `dF`, `dτ` are scaled by k (e.g. a sluggish engine).
**5f – Multiple faults:** combination of two faults at the same time (e.g. 5a + 5c), as an explicit list in the config.

---

## 6. Fault levels ("from when on is it a fault")

**6a – Fixed levels:** list in the config, e.g. `levels: [1.0, 0.5, 0.2]`. Every fault case is checked for every level.
**6b – Threshold search:** bisection over k ∈ [0, 1] per fault case and initial state. The result is the smallest residual factor `k_min` with which landing still works (≈ 6–8 solves per case at 1 % accuracy).
**6c – Both:** fixed levels for overview tables, threshold search for the key result.

> **Note on the numbers:** With m = 1 kg and `Fz_max = 20 N` the thrust-to-weight ratio is ≈ 12. Only below `k ≈ 1.62/20 ≈ 8 %` can the vehicle no longer hover. **With the current parameters, 50 % and 20 % are very likely always landable.** The interesting threshold lies at T/W ≈ 1–2 (k ≈ 8–16 %), or mass/thrust need to be chosen more realistically. This argues for 6b/6c or for finer levels.

---

## 7. Time horizon

Currently fixed: `N = 200`, `dt = 0.05 s`, i.e. T = 10 s. Landing must happen exactly at `t_N`.

**7a – Fixed:** T stays fixed (configurable in the config). Simple and robust, but "not landable" can also mean that T was chosen badly (too short, or too long so that the vehicle "uses up" propellant or altitude).
**7b – Free final time:** `T` is a decision variable, `dt = T/N` in the integrator, with `T ∈ [T_min, T_max]`. Physically the cleanest answer to "landable or not"; harder for IPOPT and needs a good initial guess.
**7c – Grid over T:** several fixed T (e.g. 5, 10, 20, 40 s). Landed if at least one of them is landable. More robust than 7b but costs more solves; each T needs its own solver (or N fixed and dt varied).

---

## 8. Fault timing

**8a – From t = 0:** `s0` already is the state at the moment of the fault. Simple; initial states come directly from the config or the sampling.
**8b – Two-stage:** solve a nominal OCP without fault to the landing zone, take the state at `t_fault` from `S_opt` and run the fault check from there with the remaining horizon time. Realistic: the fault occurs mid-approach and is not known to the planner beforehand.
**8c – Two-stage with sweep over `t_fault`:** like 8b, but for several fault times. Result e.g.: "From which phase of the approach on can a 10 % thrust fault no longer be recovered?"

---

## 9. Cost function

**9a – Keep tracking:** `Σ eᵀQe + uᵀRu` with the zone center as target, plus the hard landing constraints. Closest to the existing code; the Q weights strongly influence the trajectory, though.
**9b – Pure feasibility:** only a small regularisation `Σ uᵀRu` (smooth control). The question is solely "does a feasible trajectory exist". Fits "landable yes/no" best.
**9c – Propellant/effort-minimal:** `Σ ‖F_k‖ · dt` (∝ propellant at constant Isp, cf. README) or `Σ ‖F_k‖²` for smoothness. Additionally shows "how expensive" the emergency landing is.
**9d – Minimum time:** `min T`. **Only with 7b.**

---

## 10. Evaluation / output

**10a – Single check:** `main_landing.py` with one case from the config. Console output (landed yes/no + metrics from section 0), plots and animation with landing zone.
**10b – 10a + batch:** `main_landing_batch.py` based on `main_multiprocessing.py`. Grid or sampling over initial states (altitude, lateral offset, velocity) × fault cases × levels. `log.csv` additionally gets `fault`, `level`, `landed`, final errors and cost (these are missing there so far).
**10c – 10b + feasibility maps:** evaluation script for maps, e.g. initial position (x, z) colour-coded by landable / not landable or by `k_min` (with 6b), one map per fault case.

---

## Compatibility

| Combination | Status |
|---|---|
| 2c without 1d | not meaningful |
| 9d without 7b | not possible |
| 5b with 4b | not applicable (no lateral thrust) |
| 1b / 1c with 5b (Fy), 5c (τx, τz) | not applicable (axis does not exist in 2D) |
| 1b / 1c with 3.5, 3.7 | not applicable (no yaw in 2D); 3.3 and 3.6 for one axis only |
| 3.5b/3.5c with 5c (τz) | allowed, but τz failure with `ψ_0 ≠ ψ_t` practically always "not landable" |
| 3.1b without 3.8b | not meaningful (cutoff above ground means engine off) |
| 3.10b with 3.8 / 3.9 | 3.8/3.9 become ineffective |
| 3.10b with 5a | distorted (hovering at the ground costs thrust) |
| 3.0b with 9b | penalty terms from 3 are added to the regularisation |
| 1d with 10c | possible, maps then in arc length instead of x |
| 6b with 7b | possible, but many difficult solves; 7c is more robust |
| 8b/8c | nominal OCP uses the same class without fault |

---

## Open numerical values (with proposal, adjust after discussion)

| Quantity | Proposal | Remark |
|---|---|---|
| `landing.radius` | 1.0 m | size open, passed as parameter |
| `landing.center` | [0, 0] | |
| `v_z,max` | 1.0 m/s | 3.2b |
| `v_h,max` | 0.5 m/s | 3.3b/3.3c |
| `θ_max` | 10° | 3.4b |
| `ψ_t` / `Δψ` / n legs | 0° / 15° / 4 | 3.5b–d |
| `ω_max` (tilt) | 5 °/s | 3.6b |
| `r_max` (yaw) | 10 °/s | 3.7b |
| `τ_rest` | 0.05 Nm | 3.9c |
| `h_cut` | 0.5 m | 3.1b |
| M (hold phase) | 5 nodes | 3.10b |
| Penalty weights | – | 3.0b/3.0c, tune during implementation |
| T (or `T_min`, `T_max`) | 10 s (or 2–40 s) | 7a/7b; grid for 7c |
| Fault levels k | [1.0, 0.5, 0.2, 0.1] | see note in 6 |
| Initial state range (batch) | z ∈ [1, 10] m, x, y ∈ [−5, 5] m, v = 0 | only 10b/10c |

---

## Recommendation for getting started

`1a, 2a, 3: R, 4c, 5a+5c, 6c, 7c, 8a, 9b, 10b`

This keeps the existing model. "Landable" is defined only via physically justified conditions, with a consistent final state and free yaw, so that a τz fault does not fail artificially. The comparison with/without lateral thrust shows when torque faults matter at all. The threshold search yields a meaningful key figure (`k_min`). 8b/8c, 1c/1d and 10c can be added later without restructuring the core.
