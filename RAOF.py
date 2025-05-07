import streamlit as st
import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes

st.set_page_config(layout="wide")
st.title("TRR Isoplane Geometry Simulator – RSim11 (Surface Mode)")

st.markdown("""
This simulator renders **surface isoplanes** extracted using the **Marching Cubes algorithm**, based on resonance waves in X, Y, Z.
""")

# Presets
presets = {
    "Stable Quantum Node": {"fx": 6.0, "fy": 6.0, "fz": 6.0, "px": 0, "py": 0, "pz": 0, "threshold": 0.05},
    "Decoherence Shift": {"fx": 6.0, "fy": 6.0, "fz": 6.0, "px": 45, "py": 0, "pz": 0, "threshold": 0.05},
    "Chladni Mimic": {"fx": 3.0, "fy": 4.0, "fz": 4.0, "px": 0, "py": 0, "pz": 0, "threshold": 0.1},
    "Reality Fog": {"fx": 5.5, "fy": 6.0, "fz": 6.5, "px": 90, "py": 45, "pz": 180, "threshold": 0.3},
    "Observer Disruption": {"fx": 7.0, "fy": 7.0, "fz": 7.0, "px": 90, "py": 90, "pz": 90, "threshold": 0.05},
}

selected = st.sidebar.selectbox("Choose TRR Demo Preset", list(presets.keys()))
preset = presets[selected]

domain_scale = st.sidebar.slider("Visual Grid Scale", 1.0, 30.0, 10.0, 1.0)
grid_size = st.sidebar.slider("Grid Resolution", 20, 60, 40, 5)

log_fx = st.sidebar.slider("X Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fx"], 0.1)
log_fy = st.sidebar.slider("Y Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fy"], 0.1)
log_fz = st.sidebar.slider("Z Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fz"], 0.1)
phase_x = np.radians(st.sidebar.slider("X Phase Shift (°)", 0, 360, preset["px"], 10))
phase_y = np.radians(st.sidebar.slider("Y Phase Shift (°)", 0, 360, preset["py"], 10))
phase_z = np.radians(st.sidebar.slider("Z Phase Shift (°)", 0, 360, preset["pz"], 10))
threshold = st.sidebar.slider("Isoplane Threshold", 0.0, 1.0, preset["threshold"], 0.01)

fx, fy, fz = 10**log_fx, 10**log_fy, 10**log_fz

x = np.linspace(0, domain_scale, grid_size)
y = np.linspace(0, domain_scale, grid_size)
z = np.linspace(0, domain_scale, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Field generation
EX = np.sin(fx * np.pi * X + phase_x)
EY = np.sin(fy * np.pi * Y + phase_y)
EZ = np.sin(fz * np.pi * Z + phase_z)
interference = np.abs(EX * EY * EZ)

# Normalize field
field_norm = (interference - np.min(interference)) / (np.max(interference) - np.min(interference))
st.write("Marching on scalar field with shape:", field_norm.shape)

# Run marching cubes
try:
    verts, faces, _, _ = marching_cubes(field_norm, level=threshold)
    verts *= domain_scale / grid_size  # scale to match physical space
    x, y, z = verts.T
    i, j, k = faces.T

    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='cyan',
        opacity=0.6
    )])
    fig.update_layout(scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
        bgcolor="black"
    ), paper_bgcolor="black", margin=dict(l=0, r=0, t=0, b=0), height=800)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Surface extraction failed: {e}")

st.markdown(f"**Threshold**: {threshold:.2f} — Frequencies: X=10^{log_fx:.1f}Hz, Y=10^{log_fy:.1f}Hz, Z=10^{log_fz:.1f}Hz")
