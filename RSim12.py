
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("RAO-Triggered Photon Emergence Simulator (TRR Prototype)")

st.markdown("This simulation models local photon emergence in a nanophotonic system using TRR principles. A waveguide field intersects with a quantum emitter's localized resonance mode. Where field overlap exceeds the resonance threshold, a photon is 'rendered'.")

# --- Domain setup ---
grid_size = st.sidebar.slider("Grid Resolution", 20, 60, 40, 5)
domain_scale = st.sidebar.slider("Domain Size (microns)", 0.1, 5.0, 1.0, 0.1)
x = np.linspace(0, domain_scale, grid_size)
y = np.linspace(0, domain_scale, grid_size)
z = np.linspace(0, domain_scale, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# --- Waveguide field (external Φ) ---
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
Phi = EX * EY * EZ  # External field

# --- Quantum emitter field Ψᵣ(x,t) ---
st.sidebar.markdown("### Emitter Location")
cx = st.sidebar.slider("X Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
cy = st.sidebar.slider("Y Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
cz = st.sidebar.slider("Z Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
width = st.sidebar.slider("Emitter Width (μm)", 0.01, 1.0, 0.2, 0.01)

Ψr = np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / width**2)

# --- Resonance interaction ---
Hres = Ψr * Phi
render_energy = np.abs(Hres)**2

threshold = st.sidebar.slider("Render Threshold", 0.0, 1.0, 0.05, 0.01)
photon_mask = render_energy > threshold

xv, yv, zv = X[photon_mask], Y[photon_mask], Z[photon_mask]
color_vals = render_energy[photon_mask]

# --- Plot ---
if len(xv) > 0:
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
    fig.update_layout(
        scene=dict(
            xaxis_title="X (μm)", yaxis_title="Y (μm)", zaxis_title="Z (μm)"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black',
        scene_bgcolor='black'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No photon events detected. Try adjusting field alignment or lowering the threshold.")

st.markdown("**Interpretation**: Points shown represent local zones where resonance between emitter and field exceeds the energy threshold for photon realization, per TRR.")
