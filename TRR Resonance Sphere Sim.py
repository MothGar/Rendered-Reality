import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("TRR 3D Resonant Field: Spherical Wavefronts with Reflection")

# --- Parameters ---
st.sidebar.header("Simulation Controls")
c = 1.0  # Wave speed (normalized units)
R = st.sidebar.slider("Aluminum Sphere Radius (R)", 1.0, 10.0, 5.0)
frequency = st.sidebar.slider("RAO Frequency (Hz)", 0.5, 10.0, 2.0)
duration = st.sidebar.slider("Simulation Duration (s)", 1.0, 10.0, 5.0)
frames = st.sidebar.slider("Frames (Time Steps)", 10, 200, 50)
iso_threshold = st.sidebar.slider("Iso-Surface Threshold", 0.0, 2.0, 0.5, 0.01)

# --- Grid Setup ---
grid_res = 50
domain_size = R * 1.2
x = np.linspace(-domain_size, domain_size, grid_res)
y = np.linspace(-domain_size, domain_size, grid_res)
z = np.linspace(-domain_size, domain_size, grid_res)
X, Y, Z = np.meshgrid(x, y, z)
r = np.sqrt(X**2 + Y**2 + Z**2)

# --- Time Discretization ---
t_vals = np.linspace(0, duration, frames)
k = 2 * np.pi * frequency

# --- Field Simulation with Reflection ---
fields = []
phases = []
for t in t_vals:
    outgoing = np.sin(k * (r - c * t)) / (r + 1e-6)
    reflected = np.sin(k * (r + c * t)) / (r + 1e-6)
    total_wave = outgoing + reflected
    total_wave[r >= R] = 0
    fields.append(total_wave)
    phases.append(np.angle(np.exp(1j * k * (r - c * t))))

# --- Visualization Controls ---
st.sidebar.markdown("---")
selected_frame = st.sidebar.slider("View Frame", 0, frames - 1, 0)
view_mode = st.sidebar.radio("View Mode", ["Amplitude Slice", "Phase Map Slice", "Iso-Surface View"])

field = fields[selected_frame]
phase = phases[selected_frame]
slice_z = field[:, :, grid_res // 2]
slice_phase = phase[:, :, grid_res // 2]

if view_mode == "Amplitude Slice":
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
        title="Central XY Slice of Resonant Field (Amplitude)",
        xaxis_title="X",
        yaxis_title="Y"
    )
    st.plotly_chart(fig)

elif view_mode == "Phase Map Slice":
    fig = go.Figure(data=go.Heatmap(
        z=slice_phase,
        x=x,
        y=y,
        colorscale='twilight',
        colorbar=dict(title='Phase (rad)')
    ))
    fig.update_layout(
        width=700,
        height=700,
        title="Central XY Slice of Field Phase",
        xaxis_title="X",
        yaxis_title="Y"
    )
    st.plotly_chart(fig)

elif view_mode == "Iso-Surface View":
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(field, level=iso_threshold, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        colorscale='Viridis',
        intensity=verts[:, 2],
        showscale=True
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        width=700,
        height=700,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Iso-Surface of Resonant Field"
    )
    st.plotly_chart(fig)

st.markdown("""
This dynamic simulation shows RAO-triggered spherical wavefronts with reflection modeled inside a reflective aluminum shell. 
Use the sidebar to switch between amplitude slices, phase maps, and full 3D iso-surface visualizations.
""")
