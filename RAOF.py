import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("Resonant Standing Wave Spheres")

# --- Physical Constants ---
c = 3e8  # Speed of light (m/s)

# --- Resonant Standing Wave Generator ---
def standing_wave_sphere(center, freq, phase, radius=30):
    # Wavelength from frequency (lambda = c / f)
    wavelength = c / freq
    X, Y, Z = np.mgrid[-radius:radius:40j, -radius:radius:40j, -radius:radius:40j]
    r = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    wave = np.sin((2 * np.pi * r / wavelength) + np.radians(phase))
    return X, Y, Z, wave

# --- Sidebar Configuration ---
st.sidebar.header("Sphere Settings")
freq = st.sidebar.slider("Frequency (Hz)", 1e6, 1e12, 3e9)  # GHz range
phase = st.sidebar.slider("Phase (degrees)", 0, 360, 0)
radius = st.sidebar.slider("Sphere Radius", 10, 60, 30)

# --- Generate Resonant Sphere ---
X, Y, Z, wave = standing_wave_sphere(center=[0, 0, 0], freq=freq, phase=phase, radius=radius)

# --- Visualization ---
fig = go.Figure()

# Add the standing wave pattern as an isosurface
fig.add_trace(go.Isosurface(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=wave.flatten(),
    isomin=-0.5,  # Central wave node
    isomax=0.5,   # Positive wave node
    surface_count=5,
    opacity=0.7,
    colorscale="Blues",
    caps=dict(x_show=False, y_show=False, z_show=False),
    name="Resonant Sphere"
))

# Plot layout adjustments
fig.update_layout(scene=dict(
    aspectmode="cube",
    xaxis=dict(range=[-radius, radius]),
    yaxis=dict(range=[-radius, radius]),
    zaxis=dict(range=[-radius, radius])
), title="Resonant Standing Wave Sphere")

st.plotly_chart(fig, use_container_width=True)
