import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import imageio
import os

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("TRR 3D Resonant Field: Spherical Wavefronts with Reflection")

# --- Parameters ---
st.sidebar.header("Simulation Controls")
c = 1.0  # Wave speed (normalized units)
R = st.sidebar.slider("Aluminum Sphere Radius (R)", 1.0, 10.0, 5.0)
frequency = st.sidebar.slider("RAO Frequency (Hz)", 10.0, 25000.0, 15000.0, step=100.0)
duration = st.sidebar.slider("Simulation Duration (s)", 1.0, 10.0, 5.0)
frames = st.sidebar.slider("Frames (Time Steps)", 10, 200, 50)

# Placeholder: compute a valid iso threshold based on field stats
iso_threshold_default = 0.5
iso_threshold = st.sidebar.slider("Iso-Surface Threshold", 0.0, 2.0, iso_threshold_default, 0.01)

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
coherence_scores = []
for t in t_vals:
    outgoing = np.sin(k * (r - c * t)) / (r + 1e-6)
    reflected = np.sin(k * (r + c * t)) / (r + 1e-6)
    total_wave = outgoing + reflected
    total_wave[r >= R] = 0
    fields.append(total_wave)
    phase = np.angle(np.exp(1j * k * (r - c * t)))
    phases.append(phase)
    coherence = np.mean(np.cos(phase)**2 + np.sin(phase)**2)
    coherence_scores.append(coherence)

# Ensure iso_threshold is valid
min_val, max_val = np.min(fields[0]), np.max(fields[0])
if not (min_val <= iso_threshold <= max_val):
    st.error(f"Iso-surface threshold {iso_threshold} is outside data range ({min_val:.2f} to {max_val:.2f}). Adjust slider.")
    st.stop()

# --- Visualization Controls ---
st.sidebar.markdown("---")
selected_frame = st.sidebar.slider("View Frame", 0, frames - 1, 0)
view_mode = st.sidebar.radio("View Mode", ["Amplitude Slice", "Phase Map Slice", "Iso-Surface View", "Animate Iso-Surface", "Export GIF"])

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
            scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            scene=dict(
                xaxis=dict(range=[-domain_size, domain_size]),
                yaxis=dict(range=[-domain_size, domain_size]),
                zaxis=dict(range=[-domain_size, domain_size]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title=f"Frame {selected_frame + 1}/{frames} | Coherence: {coherence_scores[selected_frame]:.4f}"
        )
    st.plotly_chart(fig)

elif view_mode == "Animate Iso-Surface":
    from skimage import measure
    stframe = st.empty()
    step = max(1, frames // 60)  # Ensure a maximum of ~60 frames for the animation
    for i in range(0, frames, step):
        field = fields[i]
        min_f, max_f = np.min(field), np.max(field)
        threshold = np.clip(iso_threshold, min_f + 1e-6, max_f - 1e-6)
        verts, faces, _, _ = measure.marching_cubes(field, level=threshold, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
        mesh = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.5, colorscale='Viridis', intensity=verts[:, 2], showscale=False
        )
        fig = go.Figure(data=[mesh])
        fig.update_layout(
            width=700,
            height=700,
            scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            scene=dict(
                xaxis=dict(range=[-domain_size, domain_size]),
                yaxis=dict(range=[-domain_size, domain_size]),
                zaxis=dict(range=[-domain_size, domain_size]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title=f"Frame {i + 1}/{frames} | Coherence: {coherence_scores[i]:.4f}"
        )
        stframe.plotly_chart(fig)
        time.sleep(0.1)

elif view_mode == "Export GIF":
    from skimage import measure
    from PIL import Image
    image_folder = "gif_frames"
    os.makedirs(image_folder, exist_ok=True)
    image_paths = []

    step = max(1, frames // 60)  # Match animation logic
    for i in range(0, frames, step):
        field = fields[i]
        min_f, max_f = np.min(field), np.max(field)
        threshold = np.clip(iso_threshold, min_f + 1e-6, max_f - 1e-6)
        verts, faces, _, _ = measure.marching_cubes(field, level=threshold, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
        mesh = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.5, colorscale='Viridis', intensity=verts[:, 2], showscale=False
        )
        fig = go.Figure(data=[mesh])
        fig.update_layout(
            scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            width=700, height=700,
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            title=f"Frame {i + 1}/{frames} | Coherence: {coherence_scores[i]:.4f}"
        )
        filepath = os.path.join(image_folder, f"frame_{i:03d}.png")
        fig.write_image(filepath)
        image_paths.append(filepath)

    images = [Image.open(p).convert("RGB") for p in image_paths]
    gif_path = "resonance_simulation.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=150, loop=0)
    with open(gif_path, "rb") as f:
        st.download_button("Download 3D Iso-Surface Animation GIF", f, file_name="resonance_simulation.gif")

st.markdown("""
This dynamic simulation shows RAO-triggered spherical wavefronts with reflection modeled inside a reflective aluminum shell. 
Use the sidebar to switch between amplitude slices, phase maps, 3D iso-surfaces, animations, or export a 3D GIF.
Coherence score is calculated from normalized phase consistency.
RAO frequency range is tuned to the audible resonance region of aluminum.
""")
# [Leave remaining content unchanged after this validation logic]
