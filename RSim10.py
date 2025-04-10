import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def recommended_grid_size(frequencies_hz, domain_scale_m, target_grid=40, reference_freq_log10=6.0, reference_domain=1.0, min_pts=10, max_pts=100):
    c = 299_792_458  # m/s
    smallest_lambda = min(c / f for f in frequencies_hz if f > 0)
    cycles = domain_scale_m / smallest_lambda

    ref_freq = 10 ** reference_freq_log10
    ref_lambda = c / ref_freq
    ref_cycles = reference_domain / ref_lambda

    scaling_factor = cycles / ref_cycles
    grid_points = int(target_grid * scaling_factor)

    return max(min_pts, min(grid_points, max_pts))


def chladni_mode_to_waveparams(r: int, l: int, axis: str):
    base_log_freq = 6.0
    axis_shift = {'x': 0.0, 'y': 0.1, 'z': 0.2}[axis]
    freq = base_log_freq + 0.3 * r + axis_shift
    phase = (l * 90) % 360
    return freq, phase


# Chladni presets
fx1, px1 = chladni_mode_to_waveparams(2, 3, 'x')
fy1, py1 = chladni_mode_to_waveparams(2, 3, 'y')
fz1, pz1 = chladni_mode_to_waveparams(2, 3, 'z')

fx2, px2 = chladni_mode_to_waveparams(1, 2, 'x')
fy2, py2 = chladni_mode_to_waveparams(1, 2, 'y')
fz2, pz2 = chladni_mode_to_waveparams(2, 3, 'z')

presets = {
    "Toroidal Helix (r=2, l=3)": {
        "fx": fx1, "fy": fy1, "fz": fz1,
        "px": px1, "py": py1, "pz": pz1,
        "threshold": 0.5, "lock": 0.02,
        "grid_size": 50, "domain_scale": 12.0,
        "desc": "A twisted toroidal resonance structure — circular confinement meets spiral coherence."
    },
    "Axial Helix (r=1, l=2)": {
        "fx": fx2, "fy": fy2, "fz": fz2,
        "px": px2, "py": py2, "pz": pz2,
        "threshold": 0.45, "lock": 0.015,
        "grid_size": 48, "domain_scale": 10.0,
        "desc": "A biologically inspired axial helix — suitable for modeling wave-based structures like DNA cores."
    }
}

selected = st.sidebar.selectbox("Chladni Preset", list(presets.keys()))
p = presets[selected]

use_chladni = st.sidebar.checkbox("Enable Chladni Mode Input")

if use_chladni:
    r_x = st.sidebar.slider("Radial Mode rₓ", 0, 4, 2)
    l_x = st.sidebar.slider("Angular Mode lₓ", 0, 4, 3)
    r_y = st.sidebar.slider("Radial Mode rᵧ", 0, 4, 2)
    l_y = st.sidebar.slider("Angular Mode lᵧ", 0, 4, 3)
    r_z = st.sidebar.slider("Radial Mode r𝓏", 0, 4, 2)
    l_z = st.sidebar.slider("Angular Mode l𝓏", 0, 4, 3)

    log_fx, phase_x_deg = chladni_mode_to_waveparams(r_x, l_x, 'x')
    log_fy, phase_y_deg = chladni_mode_to_waveparams(r_y, l_y, 'y')
    log_fz, phase_z_deg = chladni_mode_to_waveparams(r_z, l_z, 'z')

    fx = 10 ** log_fx
    fy = 10 ** log_fy
    fz = 10 ** log_fz

    phase_x = np.radians(phase_x_deg)
    phase_y = np.radians(phase_y_deg)
    phase_z = np.radians(phase_z_deg)

    grid_size = recommended_grid_size([fx, fy, fz], p["domain_scale"])
    threshold = p["threshold"]
    lock_strength = p["lock"]
    domain_scale = p["domain_scale"]
else:
    log_fx = st.sidebar.slider("X Frequency (log₁₀ Hz)", -1.0, 17.0, value=p["fx"], step=0.1)
    log_fy = st.sidebar.slider("Y Frequency (log₁₀ Hz)", -1.0, 17.0, value=p["fy"], step=0.1)
    log_fz = st.sidebar.slider("Z Frequency (log₁₀ Hz)", -1.0, 17.0, value=p["fz"], step=0.1)

    fx = 10**log_fx
    fy = 10**log_fy
    fz = 10**log_fz

if not use_chladni:
    phase_x = np.radians(st.sidebar.slider("X-Axis Phase (°)", 0, 360, value=p["px"], step=10))
    phase_y = np.radians(st.sidebar.slider("Y-Axis Phase (°)", 0, 360, value=p["py"], step=10))
    phase_z = np.radians(st.sidebar.slider("Z-Axis Phase (°)", 0, 360, value=p["pz"], step=10))
else:
    st.sidebar.markdown("*Phase sliders are disabled — phase set from Chladni mode: l × 90°*")


    grid_size = st.sidebar.slider("Grid Resolution", 20, 100, value=p["grid_size"], step=5)
    threshold = st.sidebar.slider("Render Threshold", 0.0, 1.0, value=p["threshold"], step=0.01)
    lock_strength = st.sidebar.slider("Resonance Lock Range", 0.0, 1.0, value=p["lock"], step=0.005)
    domain_scale = st.sidebar.slider("Domain Size", 1.0, 30.0, value=p["domain_scale"], step=1.0)

# --- Safety fallback if values are not set above ---
if 'domain_scale' not in locals():
    domain_scale = 1.0

if 'grid_size' not in locals():
    grid_size = 40

if 'threshold' not in locals():
    threshold = 0.05

if 'lock_strength' not in locals():
    lock_strength = 0.01
x = np.linspace(-domain_scale/2, domain_scale/2, grid_size)
y = np.linspace(-domain_scale/2, domain_scale/2, grid_size)
z = np.linspace(-domain_scale/2, domain_scale/2, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

EX = np.sin(fx * np.pi * X + phase_x)
EY = np.sin(fy * np.pi * Y + phase_y)
EZ = np.sin(fz * np.pi * Z + phase_z)
interference = np.abs(EX * EY * EZ)

field_norm = (interference - interference.min()) / (interference.max() - interference.min())
lock_mask = ((field_norm > threshold - lock_strength) & (field_norm < threshold + lock_strength))

xv, yv, zv = X[lock_mask], Y[lock_mask], Z[lock_mask]
color_vals = field_norm[lock_mask]

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
    mode='markers',
    marker=dict(size=2, color=color_vals, colorscale='Viridis', opacity=0.6)
))
fig.update_layout(
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
    margin=dict(l=0, r=0, b=0, t=0),
    paper_bgcolor='black', scene_bgcolor='black', height=700
)
st.plotly_chart(fig, use_container_width=True)
