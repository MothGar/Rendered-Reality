import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Physical constants for plasma frequency
e = 1.602e-19       # Elementary charge (C)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
m_e = 9.109e-31     # Electron mass (kg)

# --- Generalized TRR Field Generator ---
def generate_field(center, freq, phase, grid, radius=60, mode="radial", helicity=6.0, kvec=None):
    X, Y, Z = grid
    cx, cy, cz = center
    phase_rad = np.radians(phase)

    # Convert frequency to radians per second
    omega = 2 * np.pi * freq

    if mode == "radial":
        r = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) + 1e-5
        wave = np.sin(omega * r + phase_rad)

    elif mode == "linear":
        if kvec is None:
            kvec = np.array([1.0, 0.0, 0.0])
        kx, ky, kz = kvec
        kdotr = kx * X + ky * Y + kz * Z
        wave = np.sin(omega * kdotr + phase_rad)

    elif mode == "helical":
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        theta = np.arctan2(dy, dx)
        helix_phase = helicity * theta + omega * dz
        wave = np.sin(helix_phase + phase_rad)

    else:
        wave = np.zeros_like(X)

    decay = np.exp(-(((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / radius**2))
    return decay * wave

# --- Frequency Range Selection ---
def get_frequency_range(range_type):
    ranges = {
        "Low (0.01 Hz - 1 kHz)": (0.01, 1e3),
        "Mid (1 kHz - 1 MHz)": (1e3, 1e6),
        "High (1 MHz - 1 GHz)": (1e6, 1e9),
        "Ultra-High (1 GHz - 1 THz)": (1e9, 1e12),
        "Extreme (1 THz - 1 PHz)": (1e12, 1e15),
    }
    return ranges[range_type]

# --- Setup ---
st.set_page_config(layout="wide")
st.title("TRR Plasma-Threshold Resonance Simulator")

# --- Grid Setup ---
grid_size = 100
extent = 60
lin = np.linspace(-extent, extent, grid_size)
X, Y, Z = np.meshgrid(lin, lin, lin, indexing='xy')

# --- Sidebar Controls ---
def sphere_controls(label):
    st.sidebar.header(label)
    x = st.sidebar.slider(f"{label} - X", -60.0, 60.0, 0.0)
    y = st.sidebar.slider(f"{label} - Y", -60.0, 60.0, 0.0)
    z = st.sidebar.slider(f"{label} - Z", -60.0, 60.0, 0.0)
    freq_range = st.sidebar.selectbox(f"{label} - Frequency Range", 
                                      ["Low (0.01 Hz - 1 kHz)", "Mid (1 kHz - 1 MHz)", 
                                       "High (1 MHz - 1 GHz)", "Ultra-High (1 GHz - 1 THz)", 
                                       "Extreme (1 THz - 1 PHz)"])
    freq_min, freq_max = get_frequency_range(freq_range)
    freq = st.sidebar.slider(f"{label} - Frequency (Hz)", freq_min, freq_max, (freq_min + freq_max) / 2)
    phase = st.sidebar.slider(f"{label} - Phase", 0, 360, 0)
    return np.array([x, y, z]), freq, phase

centerA, freqA, phaseA = sphere_controls("Sphere A")
include_B = st.sidebar.checkbox("Include Sphere B", value=True)
if include_B:
    centerB, freqB, phaseB = sphere_controls("Sphere B")

include_C = st.sidebar.checkbox("Include Sphere C", value=False)
if include_C:
    centerC, freqC, phaseC = sphere_controls("Sphere C")

view_mode = st.sidebar.radio("Viewer Mode", ["3D Points", "Isosurface"])
threshold_scale = st.sidebar.slider("Plasma Threshold Scale", 0.0, 1.0, 0.51)

# --- Compute Fields ---
fieldA = generate_field(centerA, freqA, phaseA, (X, Y, Z), 60)
if include_B:
    fieldB = generate_field(centerB, freqB, phaseB, (X, Y, Z), 60)
if include_C:
    fieldC = generate_field(centerC, freqC, phaseC, (X, Y, Z), 60)

if include_B and include_C:
    overlap = fieldA * fieldB * fieldC
elif include_B:
    overlap = fieldA * fieldB
elif include_C:
    overlap = fieldA * fieldC
else:
    overlap = fieldA

# --- Visualization ---
fig = go.Figure()

if view_mode == "3D Points":
    xv, yv, zv = X[overlap > threshold_scale], Y[overlap > threshold_scale], Z[overlap > threshold_scale]
    fig.add_trace(go.Scatter3d(x=xv.flatten(), y=yv.flatten(), z=zv.flatten(), 
                               mode='markers', marker=dict(size=2, color='cyan'), name="Rendered"))
else:
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=overlap.flatten(),
        isomin=0.1, isomax=1.0,
        opacity=0.5,
        colorscale="Viridis"
    ))

fig.update_layout(scene=dict(aspectmode="cube"),
                  margin=dict(l=0, r=0, t=60, b=0),
                  title="Plasma-Constrained Resonance Geometry")

st.plotly_chart(fig, use_container_width=True)
