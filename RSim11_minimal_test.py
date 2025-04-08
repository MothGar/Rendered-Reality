import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("TRR Isoplane Geometry Simulator – RSim11")
st.markdown("""
This simulator shows how resonance waves in **X, Y, and Z** create **cymatic patterns** that overlap to form **isoplanes**—stable zones of coherent energy in 3D.
""")

# Presets
presets = {
    "Stable Quantum Node": {"fx": 6.0, "fy": 6.0, "fz": 6.0, "px": 0, "py": 0, "pz": 0, "threshold": 0.05, "lock": 0.03},
    "Decoherence Shift": {"fx": 6.0, "fy": 6.0, "fz": 6.0, "px": 45, "py": 0, "pz": 0, "threshold": 0.05, "lock": 0.03},
    "Chladni Mimic": {"fx": 3.0, "fy": 4.0, "fz": 4.0, "px": 0, "py": 0, "pz": 0, "threshold": 0.1, "lock": 0.05},
    "Reality Fog": {"fx": 5.5, "fy": 6.0, "fz": 6.5, "px": 90, "py": 45, "pz": 180, "threshold": 0.3, "lock": 0.15},
    "Observer Disruption": {"fx": 7.0, "fy": 7.0, "fz": 7.0, "px": 90, "py": 90, "pz": 90, "threshold": 0.05, "lock": 0.01},
}

selected = st.sidebar.selectbox("Choose TRR Demo Preset", list(presets.keys()))
preset = presets[selected]

# UI sliders
domain_scale = st.sidebar.slider("Visual Grid Scale", 1.0, 30.0, 10.0, 1.0)
grid_size = st.sidebar.slider("Grid Resolution", 20, 60, 40, 5)

log_fx = st.sidebar.slider("X Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fx"], 0.1)
log_fy = st.sidebar.slider("Y Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fy"], 0.1)
log_fz = st.sidebar.slider("Z Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fz"], 0.1)
phase_x = np.radians(st.sidebar.slider("X Phase Shift (°)", 0, 360, preset["px"], 10))
phase_y = np.radians(st.sidebar.slider("Y Phase Shift (°)", 0, 360, preset["py"], 10))
phase_z = np.radians(st.sidebar.slider("Z Phase Shift (°)", 0, 360, preset["pz"], 10))
threshold = st.sidebar.slider("Isoplane Threshold", 0.0, 1.0, preset["threshold"], 0.01)
lock_strength = st.sidebar.slider("Resonance Lock Range", 0.0, 1.0, preset["lock"], 0.005)

fx, fy, fz = 10**log_fx, 10**log_fy, 10**log_fz

# Grid setup
x = np.linspace(0, domain_scale, grid_size)
y = np.linspace(0, domain_scale, grid_size)
z = np.linspace(0, domain_scale, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Field computation
EX = np.sin(fx * np.pi * X + phase_x)
EY = np.sin(fy * np.pi * Y + phase_y)
EZ = np.sin(fz * np.pi * Z + phase_z)
interference = np.abs(EX * EY * EZ)

# Normalize and apply threshold
field_norm = (interference - interference.min()) / (interference.max() - interference.min())
lock_mask = ((field_norm > threshold - lock_strength) & (field_norm < threshold + lock_strength))

# Diagnostics
st.write("Field norm range:", float(field_norm.min()), float(field_norm.max()))
st.write("Voxels passing lock mask:", int(np.sum(lock_mask)))

# Fallback if no voxels found
if np.sum(lock_mask) == 0:
    st.warning("No voxels matched lock mask. Using fallback Z-slice plane.")
    lock_mask = (np.abs(Z - Z.mean()) < (Z.max() - Z.min()) / 10)

xv, yv, zv = X[lock_mask], Y[lock_mask], Z[lock_mask]
st.write("Points to render:", len(xv.flatten()))

# Render
if len(xv) > 0:
    fig = go.Figure(data=[go.Scatter3d(
        x=xv.flatten(),
        y=yv.flatten(),
        z=zv.flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color='cyan',
            opacity=0.5
        )
    )])
    fig.update_layout(scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        bgcolor="black"
    ), paper_bgcolor="black", margin=dict(l=0, r=0, t=0, b=0), height=800)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Still no points to render.")

st.markdown(f"**Threshold**: {threshold:.2f} ± {lock_strength:.3f} — Frequencies: X=10^{log_fx:.1f}Hz, Y=10^{log_fy:.1f}Hz, Z=10^{log_fz:.1f}Hz")
