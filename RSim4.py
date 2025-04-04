
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from stpyvista import stpyvista

st.set_page_config(layout="wide")

st.title("TRR Resonance Simulation: Stable Matter Range")
st.markdown("Visualize 3D wave interference in the frequency range of stable matter (10^10–10^20 Hz).")

# Sidebar Controls
st.sidebar.title("Wave Parameters")
freq_x = st.sidebar.slider("X Wave Frequency (log10 Hz)", 10.0, 20.0, 13.0, 0.1)
phase_x = st.sidebar.slider("X Wave Phase (°)", 0, 360, 0, 10)
freq_y = st.sidebar.slider("Y Wave Frequency (log10 Hz)", 10.0, 20.0, 13.5, 0.1)
phase_y = st.sidebar.slider("Y Wave Phase (°)", 0, 360, 90, 10)
freq_z = st.sidebar.slider("Z Wave Frequency (log10 Hz)", 10.0, 20.0, 14.0, 0.1)
phase_z = st.sidebar.slider("Z Wave Phase (°)", 0, 360, 180, 10)

amplitude = st.sidebar.slider("Amplitude (a.u.)", 0.1, 5.0, 1.0, 0.1)
contour_levels = st.sidebar.multiselect("Contour Levels (a.u.)", [0.0, 0.5, 1.0, 1.5, 2.0], default=[0.0])
phase_locked = st.sidebar.checkbox("Phase-Locked Waves", value=False)

# Convert log10 frequency to actual frequency in Hz
f_x = 10 ** freq_x
f_y = 10 ** freq_y
f_z = 10 ** freq_z

# Adjust phases if phase-locked is enabled
if phase_locked:
    phase_y = phase_x
    phase_z = phase_x

# Time domain
t_ns = np.linspace(0, 10, 1000)
omega_x = 2 * np.pi * f_x
omega_y = 2 * np.pi * f_y
omega_z = 2 * np.pi * f_z

phi_x = np.radians(phase_x)
phi_y = np.radians(phase_y)
phi_z = np.radians(phase_z)

wave_x = amplitude * np.sin(omega_x * t_ns * 1e-9 + phi_x)
wave_y = amplitude * np.sin(omega_y * t_ns * 1e-9 + phi_y)
wave_z = amplitude * np.sin(omega_z * t_ns * 1e-9 + phi_z)
combined = wave_x + wave_y + wave_z

# Plot waveform view
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(t_ns, wave_x, label=f"X Wave (10^{freq_x:.1f} Hz)", alpha=0.6)
ax.plot(t_ns, wave_y, label=f"Y Wave (10^{freq_y:.1f} Hz)", alpha=0.6)
ax.plot(t_ns, wave_z, label=f"Z Wave (10^{freq_z:.1f} Hz)", alpha=0.6)
ax.plot(t_ns, combined, label="Combined Wave", color='black', linewidth=2)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("Use the sidebar sliders to manipulate log-scale frequencies and phase shifts. Observe how high-frequency resonances interact over time.")

# 3D Viewer with Frame Scrubbing
st.markdown("### 3D Spatial Waveform Representation")
st.session_state.setdefault("frame_index", 0)
total_frames = 10
frame_ns = np.linspace(0.0, 10.0, total_frames)

col_play, col_slider = st.columns([1, 5])
with col_play:
    autoplay = st.checkbox("▶ Auto Play")
with col_slider:
    frame_index = st.slider("Frame Index", 0, total_frames - 1, st.session_state.frame_index)

st.session_state.frame_index = (frame_index + 1) % total_frames if autoplay else frame_index

x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
z = np.linspace(0, 1, 50)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

T = frame_ns[st.session_state.frame_index] * 1e-9
EX = amplitude * np.sin(2 * np.pi * f_x * X + omega_x * T + phi_x)
EY = amplitude * np.sin(2 * np.pi * f_y * Y + omega_y * T + phi_y)
EZ = amplitude * np.sin(2 * np.pi * f_z * Z + omega_z * T + phi_z)
wave_3d = EX + EY + EZ

# Resonance coherence probability
freq_diff = abs(freq_x - freq_y) + abs(freq_y - freq_z) + abs(freq_x - freq_z)
phase_match = int(phase_locked)
coherence_score = max(0.0, 1.0 - (freq_diff / 3)) * (0.5 + 0.5 * phase_match)
st.markdown(f"**Resonance Coherence Probability:** {coherence_score:.2%}")

try:
    grid = pv.StructuredGrid(X, Y, Z)
    grid["WaveSum"] = wave_3d.flatten(order="F")
    plotter = pv.Plotter(off_screen=True)
    contour = grid.contour(contour_levels)
    plotter.add_mesh(contour, scalars="WaveSum", cmap="coolwarm")
    plotter.view_isometric()
    plotter.set_background("white")
    st.markdown(f"#### Frame at t = {frame_ns[st.session_state.frame_index]:.2f} ns")
    stpyvista(plotter)
except Exception as e:
    st.warning(f"3D rendering failed at frame index {frame_index}: {e}")
