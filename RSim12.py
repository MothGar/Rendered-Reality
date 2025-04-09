
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("RAO-Triggered Photon Emergence Simulator (TRR Prototype)")

st.markdown("""
This simulation models local **photon emergence** in a nanophotonic system using the **Theory of Rendered Reality (TRR)**.

An external waveguide field Φ(x,t) overlaps with a localized emitter field Ψᵣ(x,t).
Only when the **render energy** (TRR interaction) exceeds a threshold, a photon is realized.

TRR Render Condition (simplified):
> **|Ψᵣ · Φ|² > Tᵣ**
""")

# --- Domain setup ---
st.sidebar.title("🧩 Simulation Parameters")
grid_size = st.sidebar.slider("Grid Resolution", 20, 60, 40, 5)
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
Phi = EX * EY * EZ  # External field

# --- Localized Resonator Ψᵣ(x,t) ---
st.sidebar.title("🧿 Quantum Emitter (Resonator)")
cx = st.sidebar.slider("X Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
cy = st.sidebar.slider("Y Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
cz = st.sidebar.slider("Z Center (μm)", 0.0, domain_scale, domain_scale/2, 0.1)
width = st.sidebar.slider("Emitter Width (μm)", 0.01, 1.0, 0.2, 0.01)

Ψr = np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / width**2)

# --- RAO Filtering ---
st.sidebar.title("🎚️ RAO Frequency Matching")
target_freq = st.sidebar.slider("Resonance Frequency (log₁₀ Hz)", 12.0, 16.0, 13.5, 0.1)

# Simulate RAO filter by comparing average external field frequency to the emitter tuning
avg_freq = (log_fx + log_fy + log_fz) / 3
freq_match_quality = 1.0 - np.abs(avg_freq - target_freq) / 2.0  # range ~0 to 1
RAO_filtered_field = Phi * freq_match_quality  # damping mismatched fields

# --- TRR Interaction Energy ---
Hres = Ψr * RAO_filtered_field
render_energy = np.abs(Hres)**2

# --- Thresholding ---
st.sidebar.title("⚡ Photon Realization Threshold")
threshold = st.sidebar.slider("Render Threshold", 0.0, 1.0, 0.05, 0.01)
photon_mask = render_energy > threshold

# --- Toggle full resonance field ---
show_full_field = st.checkbox("🌀 Show Full Resonance Field Instead of Photon Events", value=False)

# --- Visualization ---
if show_full_field:
    st.subheader("🌀 Full Resonance Energy Field (No Threshold)")
    intensity = render_energy
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=intensity.flatten(),
        opacity=0.1,
        surface_count=12,
        colorscale='Inferno'
    ))
else:
    st.subheader("✨ Photon Events (TRR Collapse Zones)")
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
        st.warning("No photon events detected. Try adjusting emitter position, frequency match, or lowering the threshold.")
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

st.markdown("""
**Interpretation**:
- External Field = electromagnetic wave input (e.g., laser mode in waveguide)
- Emitter Field = localized mode (quantum dot, atom, or coherence node)
- Render Energy = product overlap of emitter × waveguide field
- RAO = frequency gating that damps mismatched inputs
- Photon Points = locations where reality is rendered (TRR threshold passed)

This is where **resonance becomes perception**.
""")
