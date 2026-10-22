# QUAS – Starting Point (lokal, git-ignored)

## Gesamtaufbau

```
                         config.yaml   (alle Parameter)
                              │
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
  main_sim.py            main_ocp.py           main_multiprocessing.py
  Modell-Check           1 OCP lösen           viele OCPs parallel
        │                     │                          │
        ▼                     ▼                          ▼
   Simulator            OCP_tracking              OCP_tracking × num_cpus
        └──────────┬──────────┴──────────────────────────┘
                   ▼
          Vehicle → vehicle.f(s,u) = ṡ  ── rk4_step_ca
                   │
        plotting / animation        results/run_<Zeit>/
```

- Die drei `main_*.py` sind **unabhängige Einstiege**, keiner ruft einen anderen auf.
- Zustand s (19): `[x y z | u v w | qw qx qy qz | p q r | Fx Fy Fz | τx τy τz]`, Steuerung u (6): `[dF/dt | dτ/dt]`.

---

## config.yaml

```
vehicle ──────► Vehicle            (mass, Ix, Iy, Iz)
simulation ───► Simulator, main_*  (dt=0.01, 5 s, initial/target_state: 18 Werte, Winkel in Grad)
visualization ► main_sim/main_ocp  (plot, animate)
ocp ──────────► OCP_tracking       (N=200, dt=0.05, Q, R, Kraft-/Moment-/Ratengrenzen)
batch ────────► main_multiprocessing (num_cpus, N_cases)
```

- Zahlen immer als `1.0e-3` schreiben – `1e-3` liest PyYAML als String.
- Nicht in der config: `g = 1.62` (vehicle.py), Sim-Controls (main_sim.py), Zufallsbereiche (main_multiprocessing.py).

---

## src/vehicle.py — `Vehicle`

```
config ──► __init__ (m, I, g=1.62) ──► ca_model()
                                          ├─ R(q)            Body → Welt
                                          ├─ ṗ = v,  v̇ = R·F/m − g
                                          ├─ q̇ = ½ q ⊗ ω
                                          ├─ ω̇ = I⁻¹(τ − ω×Iω)
                                          ├─ Ḟ = u[0:3],  τ̇ = u[3:6]
                                          └─► self.f(s, u) = ṡ   (CasADi-Funktion)
```

- `self.f` ist das zentrale Objekt – Simulator und OCP nutzen beide genau diese Funktion.
- Rigid Body (6 DoF), konstante Masse, kein Boden, Kraft und Moment unabhängig.

## src/simulator.py — `Simulator`

```
integrate(s, u, t0, tf):
   while t < tf:
      s = step(s, u)  ──► RK4(vehicle.f, s, u, dt=0.01)
      s = normalize_q(s)
   └─► times, states (≈501, 19, 1)
```

- u bleibt über den ganzen Aufruf **konstant** – kein Regler, keine Grenzen.
- Aktuell nur in main_sim genutzt; später nützlich als Gegenprobe für `U_opt` oder als „echtes Fahrzeug“ im MPC.

## src/ocp.py — `OCP_tracking`

```
OCP_tracking(config, vehicle.f)
   └─ __init__ ──► init_solver()          (1×, teuer)
                     ├─ F = RK4(f, dt=0.05)
                     ├─ x = [S (19×201), U (6×200)],  p = [s0, S_goal]
                     ├─ Kosten Σ eᵀQe + uᵀRu
                     ├─ g: Start, Dynamik, ‖q‖=1   lbx/ubx: F/τ- und Ratengrenzen
                     └─ self.solver = nlpsol(ipopt)
solve(s0, s_goal, x_init) ──► self.solver(...) ──► sol      (pro Fall; misst self.t_solve)
extract_sol(sol) ──► S_opt (201×19), U_opt (200×6), info
```

- Multiple Shooting: Zustände sind Variablen, Physik als Gleichungs-Constraint.
- Ziel ist ein Kosten-„Zug“, kein harter Endzustand; `s_goal` darf pro Zeitschritt variieren (→ Referenz für MPC).

---

## main_sim.py

```
config ──► Vehicle + Simulator
initial_state (Euler°) ──► rad ──► euler_to_q ──► s0 (19)
controls = [0,0,1.62,0,0,0]  (hart codiert, Z. 29)
sim.integrate(s0, controls, 0, 5 s) ──► plot_results / visualize_3d
```

- Reiner Plausibilitäts-Check des Modells, nichts wird optimiert.
- Controls sind **Raten** → Fz wächst mit 1.62 N/s (kein Schweben!).

## main_ocp.py

```
config ──► Vehicle + OCP_tracking   (Simulator wird erzeugt, aber nie benutzt)
initial_state, target_state (Euler°) ──► euler_to_q ──► s0, s_goal
ocp.solve(s0, s_goal) ──► extract_sol ──► plot_results / visualize_3d
```

- Ein Fall, Start/Ziel aus config – zum Tunen von Q, R und Grenzen.

## main_multiprocessing.py

```
Hauptprozess                                 Worker 1..4 (je eigener Prozess)
────────────                                 ────────────────────────────────
1. Ordner results/run_<Zeit>/ anlegen
   + config.yaml hineinkopieren
2. config laden
3. 100 zufällige Start/Ziel-Paare erzeugen
4. Tasks bauen: (id, s0, s_goal)
5. log.csv öffnen
6. Pool mit 4 Workern starten ───────────►  init_worker(config)  (1× pro Worker)
                                              → Vehicle + OCP_tracking bauen
                                              → in globaler Variable _worker_ocp merken
7. Tasks verteilen ──────────────────────►  solve_one_case(task)  (pro Fall)
                                              → solve + extract_sol
                                              → Ergebnis-Dict zurück
8. Pro fertigem Ergebnis: ◄──────────────
   - .npz speichern (falls gelöst)
   - Zeile in log.csv schreiben
   - Fortschritt drucken
9. Gesamtzeit ausgeben
```

- Zufällig nur Positionen (Seed 42), Rest 0, qw=1; Ergebnisse kommen unsortiert (`imap_unordered`).
- log.csv hat status/success/Zeit – Kosten & Endfehler fehlen (wären nützlich zum Sortieren).

---

## utils/integrator.py — `rk4_step_ca`

```
k1 = f(x,u), k2 = f(x+h/2·k1,u), k3 = f(x+h/2·k2,u), k4 = f(x+h·k3,u)
└─► x + h/6·(k1 + 2k2 + 2k3 + k4)
```

- Funktioniert mit Zahlen (Simulator) **und** CasADi-Symbolen (OCP).

## utils/quaternion.py

```
euler_to_q(euler)          [φ,θ,ψ] ──► [qw,qx,qy,qz]   (ZYX)
euler_to_q(state=s18)      s (18) ──► s (19)
normalize_q(state)         state[6:10] /= ‖·‖
q_to_euler(q (N,4))        ──► (N,3)   für Plots
```

- Euler nur für Ein-/Ausgabe (intuitiv), intern immer Quaternion.

## utils/plotting.py — `plot_results`

```
(time, states, controls[, s_goal]) ──► 2×4 Plots
   Zeile 1: Position | Geschwindigkeit | Kräfte | Kraftraten
   Zeile 2: Euler    | Drehraten       | Momente| Momentraten
```

- Erkennt selbst ob Sim (controls (6,)) oder OCP (controls (N,6)); vibe-coded.

## utils/animation.py — `visualize_3d`

```
states, dt ──► create_hopper_geometry (Rakete im Body-Frame)
           ──► pro Frame: R(q)·Vertices + Position ──► FuncAnimation (Echtzeit)
```

- Reine Visualisierung mit Trajektorie und Body-Achsen; vibe-coded.

---

## Sonstiges

- `requirements.txt`: numpy, casadi, matplotlib, PyYAML → `pip install -r requirements.txt`
- `.gitignore`: `__pycache__/`, `.vscode/`, `results/`, `startingpoint.md`
