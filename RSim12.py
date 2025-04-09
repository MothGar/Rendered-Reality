import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("RAO-Triggered Photon Emergence Simulator (TRR + Spectral RAO + Auto Trigger)")

st.markdown("""
This simulation models local **photon emergence** using TRR principles and true **RAO spectral filtering**.

The emitter Ψᵣ(x,t) is optionally auto-positioned at the strongest overlap point with the RAO-filtered wave field Φ(x,t), maximizing photon event probability.

TRR Render Condition (simplified):
> **|Ψᵣ · R̂(ν) Φ|² > Tᵣ**
""")

# --- Domain setup ---
st.sidebar.title("🧩 Simulation Parameters")
grid_size = st.sidebar.slider("Grid Resolution", 20, 64, 40, 4)
domain_scale = st.sidebar.slider("Domain Size (microns)", 0.1, 5.0, 1.0, 0.1)
x = np.linspace(0, domain_scale, grid_size)
y = np.linspace(0, domain_scale, grid_size)
z = np.linspace(0, domain_scale, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- External Field Φ(x,t) ---
st.sidebar.title("🌊 External Field (Waveguide)")
log_fx = st.sidebar.slider("X Wave Frequency (log₁₀ Hz)", 12.0, 16.0, 13.5, 0.1)
log_fy = st.sidebar.slider("Y Wave Frequency (log₁₀ Hz)", 12.0, 16.0, 13.5, 0.1)
log_fz = st.sidebar.slider("Z Wave Frequency (log₁₀ Hz)", 12.0, 16.0, 13.5, 0.1)
phase_x = np.radians(st.sidebar.slider("X Phase (°)", 0, 360, 0, 10))
phase_y = np.radians(st.sidebar.slider("Y Phase (°)", 0, 360, 0, 10))
phase_z = np.radians(st.sidebar.slider("Z Phase (°)", 0, 360, 0, 10))

fx, fy, fz = 10**log_fx, 10**log_fy, 10**log_fz
EX = np.sin(fx * np.pi * X + phase_x)
EY = np.sin(fy * np.pi * Y + phase_y)
EZ = np.sin(fz * np.pi * Z + phase_z)
Phi = EX * EY * EZ

# --- RAO Spectral Filtering ---
st.sidebar.title("🎚️ RAO Filtering (Fourier)")
target_freq = st.sidebar.slider("Resonance Frequency (log₁₀ Hz)", 12.0, 16.0, 13.5, 0.1)
target_wave_number = 10**target_freq * np.pi * domain_scale
Phi_k = np.fft.fftn(Phi)
kspace = np.fft.fftfreq(grid_size, d=(domain_scale / grid_size))
KX, KY, KZ = np.meshgrid(kspace, kspace, kspace, indexing='ij')
Kmag = np.sqrt((KX**2 + KY**2 + KZ**2))
filter_bandwidth = target_wave_number * 0.05
RAO_filter = np.exp(-((Kmag * 2 * np.pi - target_wave_number)**2) / (2 * filter_bandwidth**2))
Phi_filtered = np.real(np.fft.ifftn(Phi_k * RAO_filter))

# --- TRR Energy Estimate (pre-emitter) ---
auto_trigger = st.sidebar.checkbox("🔁 Auto-Trigger Emitter to Max Field", value=True)

# --- Emitter Placement ---
st.sidebar.title("🧿 Emitter Settings")
width = st.sidebar.slider("Emitter Width (μm)", 0.01, 1.0, 0.2, 0.01)
Ψr = np.ones_like(Phi_filtered)  # temporary placeholder for render energy

if auto_trigger:
    # First pass to find max render energy zone
    Hres_preview = np.abs(Phi_filtered)
    max_loc = np.unravel_index(np.argmax(Hres_preview), Hres_preview.shape)
    cx, cy, cz = X[max_loc], Y[max_loc], Z[max_loc]
    st.sidebar.markdown(f"**Auto-Centered at:** ({cx:.2f}, {cy:.2f}, {cz:.2f}) μm")
else:
    cx = st.sidebar.slider("X Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
    cy = st.sidebar.slider("Y Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
    cz = st.sidebar.slider("Z Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)

# Emitter profile
Ψr = np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / width**2)

# --- TRR Interaction ---
Hres = Ψr * Phi_filtered
render_energy = np.abs(Hres)**2
render_energy = (render_energy - render_energy.min()) / (render_energy.max() - render_energy.min())  # normalize

# --- Thresholding ---
st.sidebar.title("⚡ Photon Threshold")
threshold = st.sidebar.slider("Render Threshold", 0.0, 1.0, 0.05, 0.01)
photon_mask = render_energy > threshold

# --- Full field toggle ---
show_full_field = st.checkbox("🌀 Show Full Resonance Field Instead of Photon Events", value=False)

# --- Plotting ---
if show_full_field:
    st.subheader("🌀 Full Resonance Energy Field (RAO + Emitter)")
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=render_energy.flatten(),
        opacity=0.1,
        surface_count=12,
        colorscale='Inferno'
    ))
else:
    st.subheader("✨ Photon Events (TRR Collapse)")
    if np.count_nonzero(photon_mask) > 0:
        xv, yv, zv = X[photon_mask], Y[photon_mask], Z[photon_mask]
        color_vals = render_energy[photon_mask]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
            mode='markers',
            marker=dict(
                size=3,
                color=color_vals,
                colorscale='Inferno',
                opacity=0.8,
            )
        ))
    else:
        st.warning("No photon events found. Try lowering threshold or adjusting frequencies.")
        fig = go.Figure()

fig.update_layout(
    scene=dict(
        xaxis_title="X (μm)", yaxis_title="Y (μm)", zaxis_title="Z (μm)"
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    paper_bgcolor='black',
    scene_bgcolor='black'
)
st.plotly_chart(fig, use_container_width=True)
