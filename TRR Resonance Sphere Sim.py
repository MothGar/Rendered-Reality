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

# Continue with visualizations...
# [Leave remaining content unchanged after this validation logic]
