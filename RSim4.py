import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("TRR Resonance Simulation – 1D Wave Interference")
st.markdown("This app visualizes wave interference across X, Y, Z axes in the stable matter frequency range (10^10–10^20 Hz). Use the sidebar to adjust parameters.")

# Sidebar Controls
st.sidebar.title("Wave Parameters")
freq_x = st.sidebar.slider("X Wave Frequency (log10 Hz)", 10.0, 20.0, 13.0, 0.1)
phase_x = st.sidebar.slider("X Wave Phase (°)", 0, 360, 0, 10)
freq_y = st.sidebar.slider("Y Wave Frequency (log10 Hz)", 10.0, 20.0, 13.5, 0.1)
phase_y = st.sidebar.slider("Y Wave Phase (°)", 0, 360, 90, 10)
freq_z = st.sidebar.slider("Z Wave Frequency (log10 Hz)", 10.0, 20.0, 14.0, 0.1)
phase_z = st.sidebar.slider("Z Wave Phase (°)", 0, 360, 180, 10)

amplitude = st.sidebar.slider("Amplitude (a.u.)", 0.1, 5.0, 1.0, 0.1)
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

# Coherence probability
freq_diff = abs(freq_x - freq_y) + abs(freq_y - freq_z) + abs(freq_x - freq_z)
phase_match = int(phase_locked)
coherence_score = max(0.0, 1.0 - (freq_diff / 3)) * (0.5 + 0.5 * phase_match)
st.markdown(f"**Resonance Coherence Probability:** {coherence_score:.2%}")
