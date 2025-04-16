# TRR Full Simulator: Interactive 3D Overlap + Wave Interference Panel + 3D Isoplane Viewer with 3 Spheres

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- Generate 3D Resonance Field ---
def generate_field(center, freq, phase, grid, radius=60):
    X, Y, Z = grid
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    decay = np.exp(-((dist / radius)**2))
    wave = np.sin(freq * dist + np.radians(phase))
    return decay * wave

# --- Generate Side-View Overlapping Waves ---
def generate_overlapping_waves(freq1, phase1_deg, freq2, phase2_deg, extent=60, resolution=1000):
    x = np.linspace(-extent, extent, resolution)
    phase1_rad = np.radians(phase1_deg)
    phase2_rad = np.radians(phase2_deg)
    wave1 = np.sin(freq1 * x + phase1_rad)
    wave2 = np.sin(freq2 * x + phase2_rad)
    product = wave1 * wave2
    return x, wave1, wave2, product

# --- Setup ---
st.set_page_config(layout="wide")
st.title("TRR Full Simulator — Render Fields, Wave Interference, and Isoplane")

# --- Grid Setup ---
grid_size = 100
extent = 60
lin = np.linspace(-extent, extent, grid_size)
X, Y, Z = np.meshgrid(lin, lin, lin)

# --- Sidebar Controls ---
st.sidebar.header("Sphere A")
xA = st.sidebar.slider("A - X Pos", -60.0, 60.0, -20.0, step=1.0)
yA = st.sidebar.slider("A - Y Pos", -60.0, 60.0, 0.0, step=1.0)
zA = st.sidebar.slider("A - Z Pos", -60.0, 60.0, 0.0, step=1.0)
freqA = st.sidebar.slider("A - Frequency", 0.1, 5.0, 0.1, step=0.01)
phaseA = st.sidebar.slider("A - Phase (°)", 0, 360, 0, step=5)

st.sidebar.header("Sphere B")
xB = st.sidebar.slider("B - X Pos", -60.0, 60.0, 20.0, step=1.0)
yB = st.sidebar.slider("B - Y Pos", -60.0, 60.0, 0.0, step=1.0)
zB = st.sidebar.slider("B - Z Pos", -60.0, 60.0, 0.0, step=1.0)
freqB = st.sidebar.slider("B - Frequency", 0.1, 5.0, 0.1, step=0.01)
phaseB = st.sidebar.slider("B - Phase (°)", 0, 360, 90, step=5)

st.sidebar.header("Sphere C (Observer)")
xC = st.sidebar.slider("C - X Pos", -60.0, 60.0, 0.0, step=1.0)
yC = st.sidebar.slider("C - Y Pos", -60.0, 60.0, 30.0, step=1.0)
zC = st.sidebar.slider("C - Z Pos", -60.0, 60.0, 0.0, step=1.0)
freqC = st.sidebar.slider("C - Frequency", 0.1, 5.0, 0.1, step=0.01)
phaseC = st.sidebar.slider("C - Phase (°)", 0, 360, 180, step=5)

threshold = st.sidebar.slider("Render Threshold", 0.05, 1.0, 0.5, step=0.01)
include_C = st.sidebar.checkbox("Include Observer C in Calculation", value=True)
view_mode = st.sidebar.radio("Viewer Mode", ["3D Render", "3D Isoplane View"])

# --- Compute Fields ---
centerA = np.array([xA, yA, zA])
centerB = np.array([xB, yB, zB])
centerC = np.array([xC, yC, zC])
radius = 60

fieldA = generate_field(centerA, freqA, phaseA, (X, Y, Z), radius)
fieldB = generate_field(centerB, freqB, phaseB, (X, Y, Z), radius)
fieldC = generate_field(centerC, freqC, phaseC, (X, Y, Z), radius)

overlap = fieldA * fieldB * fieldC if include_C else fieldA * fieldB
render_zone = np.abs(overlap) > threshold

# --- Viewer Toggle ---
if view_mode == "3D Render":
    xv, yv, zv = X[render_zone], Y[render_zone], Z[render_zone]
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=xv.flatten(), y=yv.flatten(), z=zv.flatten(), mode='markers', marker=dict(size=2, color='lime', opacity=0.5), name="Rendered Zone"))
    fig3d.add_trace(go.Scatter3d(x=[xA], y=[yA], z=[zA], mode='markers+text', marker=dict(size=8, color='blue'), text=["Sphere A"], name="Sphere A"))
    fig3d.add_trace(go.Scatter3d(x=[xB], y=[yB], z=[zB], mode='markers+text', marker=dict(size=8, color='red'), text=["Sphere B"], name="Sphere B"))
    fig3d.add_trace(go.Scatter3d(x=[xC], y=[yC], z=[zC], mode='markers+text', marker=dict(size=8, color='orange'), text=["Observer C"], name="Sphere C"))
    fig3d.update_layout(scene=dict(xaxis=dict(range=[-30, 30]), yaxis=dict(range=[-30, 30]), zaxis=dict(range=[-30, 30]), aspectmode="cube"), margin=dict(l=0, r=0, t=40, b=0), title="Rendered Reality Volume (3-Sphere Overlap)")
    st.subheader("3D Rendered Overlap Zone")
    st.plotly_chart(fig3d, use_container_width=True)
else:
    fig_iso3d = go.Figure()
    isoplane_data = overlap if include_C else (fieldA * fieldB)
    fig_iso3d.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=isoplane_data.flatten(),
        isomin=threshold,
        isomax=overlap.max(),
        surface_count=1,
        opacity=0.6,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=True,
        name="Isoplane Surface"
    ))
    fig_iso3d.update_layout(scene=dict(xaxis=dict(range=[-30, 30]), yaxis=dict(range=[-30, 30]), zaxis=dict(range=[-30, 30]), aspectmode="cube"), margin=dict(l=0, r=0, t=40, b=0), title="3-Sphere Isoplane Resonance Field")
    st.subheader("3D Isoplane Field Structure")
    st.plotly_chart(fig_iso3d, use_container_width=True)

# --- Wave Panel ---
x_wave, wA, wB, wAB = generate_overlapping_waves(freqA, phaseA, freqB, phaseB)
_, _, wC, wABC = generate_overlapping_waves(freqA * 0 + freqC, 0, freqC, phaseC)
fig_wave, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axs[0].plot(x_wave, wA, color='blue', label='Wave A')
axs[0].plot(x_wave, wB, color='red', label='Wave B', linestyle='dashed')
axs[0].plot(x_wave, wC, color='orange', label='Wave C', linestyle='dotted')
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].set_title("Input Resonance Waves")
axs[1].plot(x_wave, wABC, color='green', label='Product (Render Signal)')
axs[1].axhline(0, color='gray', lw=0.5)
axs[1].set_xlabel("Position (X)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].set_title("Wave Product (Realization Field)")
axs[1].set_title("Wave Product (Realization Field)")

st.subheader("Wave Interference Viewer (Side Slice)")
st.pyplot(fig_wave)

with st.expander("Explanation"):
    st.markdown("""
    This simulation visualizes a TRR-style field overlap in multiple ways:

    - **3D Volume**: Points where Sphere A, B, and Observer C all overlap and exceed the threshold become *rendered*.
    - **3D Isoplane**: Collapse surfaces generated by triadic resonance alignment.
    - **Wave Panel**: A side-view slice showing interference patterns and the strength of their product field.

    You can use the third sphere to simulate observer tuning or add more harmonic complexity.
    """)
