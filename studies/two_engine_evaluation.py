'''
assumption: 2 main engines

compute the thrust of the main propulsion system in vertical direction depending on the engine tilt beta
further computes the torque resulting from one engine operating at X% lower thrust than the other

'''

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

### main engine
N = 2
F_max = 44e3 / N

# Force variation
F_min_factor = 0.10
F_max_factor = 1.00
isolines_step = 0.10

l_l = 0.65
l_v = np.array([0.81, 1.57]) # beginning of burn, end of burn

beta_deg = np.linspace(45, 90, 100)
beta = np.deg2rad(beta_deg)

# Force scaling factors for isolines
force_factors = np.arange(
    F_min_factor + isolines_step,
    F_max_factor,
    isolines_step
)

### rcs 
N_rcs = 4
F_rcs = 445
lever_rcs = 1.679


# ============================================================
# FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

# ============================================================
# PLOT 1: RELATIVE THRUST COMPONENT IN X DIRECTION
# ============================================================

Fx_rel = np.sin(beta)

ax1.plot(
    beta_deg,
    Fx_rel,
    color='steelblue',
    linewidth=2,
    label='Relative thrust in x direction'
)

# Reference line
ax1.axhline(
    0.95,
    color='red',
    linestyle='--',
    label='Fx = 0.95'
)

beta_095 = np.rad2deg(np.arcsin(0.95))

ax1.axvline(
    beta_095,
    color='red',
    linestyle='--',
    label=f'Beta = {beta_095:.2f} deg'
)

ax1.set_ylim(0, 1.05)
ax1.grid()
ax1.legend()
ax1.set_title('Relative part of the thrust pointing in x direction')
ax1.set_ylabel('Relative Fx [-]')


# ============================================================
# PLOT 2: MOMENT PRODUCED BY ONE ENGINE
# ============================================================

l_v, beta_grid = np.meshgrid(l_v, beta)

Fx = np.sin(beta_grid) * F_max
Fz = np.cos(beta_grid) * F_max

# Moment at 100% force
My = Fx * l_l - Fz * l_v

# Moment range
My_min = My * F_min_factor
My_max = My * F_max_factor

# RCS moment -> could counteract moment imbalance
My_rcs = N_rcs * F_rcs * lever_rcs

colors = ['tab:blue', 'tab:orange']

for i in range(2):

    # Colored band
    ax2.fill_between(
        beta_deg,
        My_min[:, i],
        My_max[:, i],
        color=colors[i],
        alpha=0.25,
        label=f'Pivot Main Engine = {l_v[0,i]:.2f} m'
    )

    # Isolines with relative thrust labels
    for factor in force_factors:

        ax2.plot(
            beta_deg,
            My[:, i] * factor,
            color=colors[i],
            alpha=0.8,
            linewidth=0.8
        )

        # Label at the right-hand end of each isoline
        ax2.annotate(
            f'{factor * 100:.0f}%',
            xy=(beta_deg[-1], My[-1, i] * factor),
            xytext=(5, 0),
            textcoords='offset points',
            color=colors[i],
            fontsize=8,
            va='center'
        )

    # Original 100% force curve
    ax2.plot(
        beta_deg,
        My[:, i],
        color=colors[i],
        linewidth=1.5
    )

    # Mean of original 100% curve
    My_mean = np.mean(My[:, i])

    idx = np.argmin(np.abs(My[:, i] - My_mean))
    beta_mean = beta_deg[idx]

    ax2.axvline(
        beta_mean,
        color=colors[i],
        linestyle='--',
        label=f'Mean pivot {l_v[0,i]:.2f} m: '
              f'Beta = {beta_mean:.2f} deg'
    )

# Labels for burn phases
ax2.text(
    beta_deg[5], 0.5 * (My_min[5, 0] + My_max[5, 0]),
    'End of burn',
    color='tab:blue',
    fontsize=10,
    fontweight='bold'
)

ax2.text(
    beta_deg[5], 0.5 * (My_min[5, 1] + My_max[5, 1]),
    'Beginning of burn',
    color='tab:orange',
    fontsize=10,
    fontweight='bold'
)

# RCS reference
ax2.axhline(
    My_rcs,
    color='red',
    linestyle='--',
    label='Max Moment by RCS'
)
ax2.axhline(
    -My_rcs,
    color='red',
    linestyle='--'
)

ax2.grid()
ax2.legend()
ax2.set_title('Moment produced by one engine')
ax2.set_xlabel('Beta [deg]')
ax2.set_ylabel('My')

# Make room for isoline labels on the right
ax2.margins(x=0.12)

plt.tight_layout()
#plt.savefig('results/moments_by_main_engine_and_rcs.png')
plt.show()