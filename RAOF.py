import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

# Streamlit setup
st.set_page_config(layout="wide")
st.title("3D Isosurface Viewer: Frequency-Driven Chladni Modes")

# Frequency Ranges
freq_ranges = {
    "Low (0.01 Hz - 1 kHz)": (0.01, 1e3),
    "Mid (1 kHz - 1 MHz)": (1e3, 1e6),
    "High (1 MHz - 1 GHz)": (1e6, 1e9),
    "Ultra-High (1 GHz - 1 THz)": (1e9, 1e12),
    "Extreme (1 THz - 1 PHz)": (1e12, 1e15),
}

def get_frequency_range(range_name):
    return freq_ranges[range_name]

# Grid resolution
grid_size = 64
lin = np.linspace(-1, 1, grid_size)
Xg, Yg, Zg = np.meshgrid(lin, lin, lin, indexing='ij')

# Convert to cylindrical coordinates for each plane
R_xy = np.sqrt(Xg**2 + Yg**2)
Theta_xy = np.arctan2(Yg, Xg)
R_yz = np.sqrt(Yg**2 + Zg**2)
Theta_yz = np.arctan2(Zg, Yg)
R_zx = np.sqrt(Zg**2 + Xg**2)
Theta_zx = np.arctan2(Xg, Zg)

# Chladni pattern generator
def chladni_pattern(R, Theta, l, r, freq, phase):
    n = r + 1
    zeros = jn_zeros(l, n)
    k_ln = zeros[-1] if len(zeros) > 0 else 1.0
    omega = 2 * np.pi * freq
    scaled_R = np.clip(R, 0, 1) * (k_ln * (freq / 1000)**0.5)
    wave = np.cos(l * Theta + omega * scaled_R + np.radians(phase)) * jn(l, scaled_R)
    return wave

# Sidebar controls
st.sidebar.title("Mode Selection and Frequency")
freq_range_name = st.sidebar.selectbox("Frequency Range", list(freq_ranges.keys()))
freq_min, freq_max = get_frequency_range(freq_range_name)

# Text input for frequency
freq_input = st.sidebar.text_input(f"Frequency (Hz) [{freq_min} - {freq_max}]", value=str((freq_min + freq_max) / 2))
try:
    freq = float(freq_input)
    if not (freq_min <= freq <= freq_max):
        st.sidebar.error(f"Please enter a frequency between {freq_min} and {freq_max}.")
        freq = (freq_min + freq_max) / 2
except ValueError:
    st.sidebar.error("Invalid input. Please enter a numeric value.")
    freq = (freq_min + freq_max) / 2

phase = st.sidebar.slider("Phase (degrees)", 0, 360, 0, step=45)
domain_size = st.sidebar.slider("Domain Display Size", 1.0, 10.0, 2.0, step=0.5)

x_index = st.sidebar.selectbox("X Axis Mode (l,r)", list(range(16)), format_func=lambda i: f"{i+1}. {modes[i]}")
y_index = st.sidebar.selectbox("Y Axis Mode (l,r)", list(range(16)), format_func=lambda i: f"{i+1}. {modes[i]}")
z_index = st.sidebar.selectbox("Z Axis Mode (l,r)", list(range(16)), format_func=lambda i: f"{i+1}. {modes[i]}")

x_l, x_r = modes[x_index]
y_l, y_r = modes[y_index]
z_l, z_r = modes[z_index]

# Compute Chladni wave on each plane with frequency and phase
Wx = chladni_pattern(R_yz, Theta_yz, x_l, x_r, freq, phase)
Wy = chladni_pattern(R_zx, Theta_zx, y_l, y_r, freq, phase)
Wz = chladni_pattern(R_xy, Theta_xy, z_l, z_r, freq, phase)

# Combine into 3D field (multiplicative model)
W = Wx * Wy * Wz

# Normalize
W = np.nan_to_num(W)
W /= np.max(np.abs(W)) + 1e-9

# Plotly isosurface rendering
fig = go.Figure(data=go.Isosurface(
    x=(Xg * domain_size).flatten(), y=(Yg * domain_size).flatten(), z=(Zg * domain_size).flatten(),
    value=W.flatten(),
    isomin=0.5, isomax=0.9,
    surface_count=4,
    opacity=0.7,
    colorscale='Plasma',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

fig.update_layout(title="3D Isosurface of Frequency-Driven Chladni Modes",
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
st.plotly_chart(fig, use_container_width=True)
