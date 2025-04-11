import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("TRR 3D Resonant Field: Spherical Wavefronts in Reflective Boundary")

# --- Parameters ---
st.sidebar.header("Simulation Controls")
c = 1.0  # Wave speed in normalized units
R = st.sidebar.slider("Aluminum Sphere Radius (R)", 1.0, 10.0, 5.0)
frequency = st.sidebar.slider("RAO Frequency (Hz)", 0.5, 10.0, 2.0)
duration = st.sidebar.slider("Simulation Duration (s)", 1.0, 10.0, 5.0)
frames = st.sidebar.slider("Frames (Time Steps)", 10, 200, 50)

# --- Grid Setup ---
grid_res = 50
domain_size = R * 1.2  # Slightly larger than sphere
x = np.linspace(-domain_size, domain_size, grid_res)
y = np.linspace(-domain_size, domain_size, grid_res)
z = np.linspace(-domain_size, domain_size, grid_res)
X, Y, Z = np.meshgrid(x, y, z)
r = np.sqrt(X**2 + Y**2 + Z**2)

# --- Time Discretization ---
t_vals = np.linspace(0, duration, frames)
k = 2 * np.pi * frequency

# --- Field Simulation ---
fields = []
for t in t_vals:
    wave = np.sin(k * (r - c * t)) / (r + 1e-6)  # Spherical wave
    wave[r >= R] = 0  # Reflective boundary condition
    fields.append(wave)

# --- Visualization: Show One Frame ---
st.sidebar.markdown("---")
selected_frame = st.sidebar.slider("View Frame", 0, frames - 1, 0)
st.write(f"Frame: {selected_frame + 1} / {frames}")

field = fields[selected_frame]
slice_z = field[:, :, grid_res // 2]  # Middle Z slice

fig = go.Figure(data=go.Heatmap(
    z=slice_z,
    x=x,
    y=y,
    colorscale='RdBu',
    zmid=0,
    colorbar=dict(title='Field Amplitude')
))
fig.update_layout(
    width=700,
    height=700,
    title="Central XY Slice of Resonant Field",
    xaxis_title="X",
    yaxis_title="Y"
)
st.plotly_chart(fig)

st.markdown("""
This is a first-pass dynamic simulation of RAO-triggered spherical wavefronts inside a reflective aluminum boundary. 
Wavefronts are cut at the spherical edge and reflections will be modeled in the next version.
""")
