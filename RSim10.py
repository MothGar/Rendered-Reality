import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_PANEL"] = "false"
import streamlit as st
import numpy as np
import pyvista as pv
from stpyvista import stpyvista

st.set_page_config(layout="wide")
st.title("TRR Isoplane Geometry Simulator – RSim10")
st.markdown("""
This simulator shows how resonance waves in **X, Y, and Z** create **cymatic patterns** that overlap to form **isoplanes**—stable zones of coherent energy in 3D. With **phase shifting** and **locking behavior**, you can see how aligned resonance forms geometries of stability.
""")

# --- Preset Definitions (log₁₀ scale based) ---
presets = {
    "Custom": {
    "Chladni Base": {'fx': 0.1, 'fy': 0.1, 'fz': 0.1, 'px': 0, 'py': 0, 'pz': 0, 'threshold': 0.5, 'lock': 0.01},
    "Chladni X Freq 1Hz": {'fx': 1.0, 'fy': 0.1, 'fz': 0.1, 'px': 0, 'py': 0, 'pz': 0, 'threshold': 0.5, 'lock': 0.01},
    "Chladni Y Freq 1Hz": {'fx': 0.1, 'fy': 1.0, 'fz': 0.1, 'px': 0, 'py': 0, 'pz': 0, 'threshold': 0.5, 'lock': 0.01},
    "Chladni Z Freq 1Hz": {'fx': 0.1, 'fy': 0.1, 'fz': 1.0, 'px': 0, 'py': 0, 'pz': 0, 'threshold': 0.5, 'lock': 0.01},
    "Chladni Phase Y 90": {'fx': 0.1, 'fy': 0.1, 'fz': 0.1, 'px': 0, 'py': 90, 'pz': 0, 'threshold': 0.5, 'lock': 0.01},
    "Chladni Phase Z 180": {'fx': 0.1, 'fy': 0.1, 'fz': 0.1, 'px': 0, 'py': 0, 'pz': 180, 'threshold': 0.5, 'lock': 0.01},
},
    "Stable Quantum Node": {"fx": 6.0, "fy": 6.0, "fz": 6.0, "px": 0, "py": 0, "pz": 0, "threshold": 0.05, "lock": 0.03},
    "Decoherence Shift": {"fx": 6.0, "fy": 6.0, "fz": 6.0, "px": 45, "py": 0, "pz": 0, "threshold": 0.05, "lock": 0.03},
    "Chladni Mimic": {"fx": 3.0, "fy": 4.0, "fz": 4.0, "px": 0, "py": 0, "pz": 0, "threshold": 0.1, "lock": 0.05},
    "Reality Fog": {"fx": 5.5, "fy": 6.0, "fz": 6.5, "px": 90, "py": 45, "pz": 180, "threshold": 0.3, "lock": 0.15},
    "Observer Disruption": {"fx": 7.0, "fy": 7.0, "fz": 7.0, "px": 90, "py": 90, "pz": 90, "threshold": 0.05, "lock": 0.01}
}

selected = st.sidebar.selectbox("Choose TRR Demo Preset", list(presets.keys()))
preset = presets[selected]

# UI and Default Configs
st.sidebar.title("Wave Parameters")
domain_scale = st.sidebar.slider("Visual Grid Scale", 1.0, 50.0, 10.0, 1.0)
invert_render = st.sidebar.checkbox("Invert Rendering (Show Negative Space)", value=False)
force_phase_lock = st.sidebar.checkbox("Phase Lock (No Phase Shift)", value=False)

grid_size = st.sidebar.slider("Grid Resolution", 20, 80, 50, 5)

# Logarithmic Frequency Control
st.sidebar.markdown("— Frequency Controls (log₁₀ Hz) —")
log_fx = st.sidebar.slider("X Frequency (log₁₀ Hz)",  -1.0, 17.0, preset.get("fx", 6.0), 0.1)
log_fy = st.sidebar.slider("Y Frequency (log₁₀ Hz)",  -1.0, 17.0, preset.get("fy", 6.0), 0.1)
log_fz = st.sidebar.slider("Z Frequency (log₁₀ Hz)",  -1.0, 17.0, preset.get("fz", 6.0), 0.1)
fx, fy, fz = 10 ** log_fx, 10 ** log_fy, 10 ** log_fz

match_freq = st.sidebar.checkbox("Auto-Scale Grid to Frequency", value=False)
if match_freq:
    grid_size = min(100, int(10 * max(log_fx, log_fy, log_fz)))

# Phase and Threshold
st.sidebar.markdown("— Phase & Threshold —")
phase_x = st.sidebar.slider("X Phase Shift (°)", 0, 360, preset.get("px", 0), 10)
phase_y = st.sidebar.slider("Y Phase Shift (°)", 0, 360, preset.get("py", 0), 10)
phase_z = st.sidebar.slider("Z Phase Shift (°)", 0, 360, preset.get("pz", 0), 10)
threshold = st.sidebar.slider("Isoplane Threshold", 0.0, 1.0, preset.get("threshold", 0.05), 0.01)
lock_strength = st.sidebar.slider("Resonance Lock Range", 0.0, 1.0, preset.get("lock", 0.03), 0.005)
use_rgb_color = st.sidebar.checkbox("Color by Wave Contribution (RGB)", value=False)

# Create Grid
x = np.linspace(0, domain_scale, grid_size)
y = np.linspace(0, domain_scale, grid_size)
z = np.linspace(0, domain_scale, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Phase shift handling
if force_phase_lock:
    px = py = pz = 0
else:
    px = np.radians(phase_x)
    py = np.radians(phase_y)
    pz = np.radians(phase_z)

# Generate Waves
EX = np.sin(fx * np.pi * X + px)
EY = np.sin(fy * np.pi * Y + py)
EZ = np.sin(fz * np.pi * Z + pz)
interference = np.abs(EX * EY * EZ)

# Normalize and lock
field_norm = (interference - np.min(interference)) / (np.max(interference) - np.min(interference))
if invert_render:
    lock_mask = (field_norm < threshold - lock_strength) | (field_norm > threshold + lock_strength)
else:
    lock_mask = (field_norm > threshold - lock_strength) & (field_norm < threshold + lock_strength)

# Visualize
volume = pv.wrap(lock_mask.astype(float))
volume.spacing = (domain_scale / grid_size, domain_scale / grid_size, domain_scale / grid_size)
plotter = pv.Plotter(off_screen=False, notebook=False)

if volume.n_points > 0:
    isoplanes = volume.contour([0.5])
    if isoplanes.n_points > 0:
        if use_rgb_color:
            rgb_field = np.stack([np.abs(EX), np.abs(EY), np.abs(EZ)], axis=-1)
            rgb_field = rgb_field / np.max(rgb_field)
            from scipy.spatial import cKDTree
            color_values = rgb_field.reshape(-1, 3)
            coords = np.array(list(volume.points))
            tree = cKDTree(coords)
            surface_coords = np.array(list(isoplanes.points))
            _, idx = tree.query(surface_coords)
            isoplanes.point_data["colors"] = color_values[idx]
            plotter.add_mesh(isoplanes, scalars=isoplanes.point_data["colors"], rgb=True, opacity=0.5)
        else:
            plotter.add_mesh(isoplanes, color="cyan", opacity=0.7)
    else:
        st.warning("No isoplane geometry formed. Adjust frequency or threshold.")
else:
    st.warning("Empty resonance mesh. Adjust parameters for better coherence.")

plotter.set_background("black")
plotter.view_isometric()
st.markdown(f"#### Frequency Range: X = 10^{log_fx:.1f} Hz | Y = 10^{log_fy:.1f} Hz | Z = 10^{log_fz:.1f} Hz")
st.markdown(f"#### Threshold: {threshold:.2f} | Lock Range ±{lock_strength:.3f}")
stpyvista(plotter)
