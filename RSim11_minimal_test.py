import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

st.set_page_config(layout="wide")
st.title("TRR Isoplane Geometry Simulator – RSim10 (Plotly Version)")

st.markdown("""
This simulator shows how resonance waves in **X, Y, and Z** create **cymatic patterns**—stable 3D geometries that emerge when waves intersect just right.
""")

# --- Preset Definitions ---
presets = {
    "Resonant Core (Ψₐ ∩ Φₐ)": {
        "desc": "A perfect match of internal and external waveforms—reality crystallizes at the center. This is the golden zone where rendering is guaranteed.",
        "fx": 6.0, "fy": 6.0, "fz": 6.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.05, "lock": 0.03,
        "grid_size": 1
    },
    "Phase Rift": {
        "desc": "One axis breaks coherence—TRR shows how a single misalignment disrupts what is rendered. Like trying to tune a radio with one knob off.",
        "fx": 6.0, "fy": 6.0, "fz": 6.0,
        "px": 45, "py": 0, "pz": 0,
        "threshold": 0.05, "lock": 0.03,
        "grid_size": 1
    },
    "Cymatic Shell": {
        "desc": "Layered wave harmonics generate cymatic-like structures. This preset mimics sound-driven geometry—where resonance creates shells of stillness.",
        "fx": 3.0, "fy": 4.0, "fz": 4.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.1, "lock": 0.05,
        "grid_size": 22
    },
    "Render Fog": {
        "desc": "Rendering is a struggle in this chaotic field. Fields are almost coherent, but never quite stabilize—like trying to see through shifting mist.",
        "fx": 5.5, "fy": 6.0, "fz": 6.5,
        "px": 90, "py": 45, "pz": 180,
        "threshold": 0.3, "lock": 0.15,
        "grid_size": 3
    },
    "Perceptual Jam": {
        "desc": "When your perception filter is 90° out of sync, nothing gets through. The system denies rendering—reality blinks out.",
        "fx": 7.0, "fy": 7.0, "fz": 7.0,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.05, "lock": 0.01,
        "grid_size": 30
    },
}

selected = st.sidebar.selectbox("Choose TRR Demo Preset", list(presets.keys()))
preset = presets[selected]

st.sidebar.markdown(f"**Description:** {preset['desc']}")

domain_scale = st.sidebar.slider("Grid Size (Visualization Scale)", 1.0, 30.0, 10.0, 1.0)
# --- Grid Size Handling ---
if "grid_size" in preset:
    grid_size = preset["grid_size"]
    st.sidebar.markdown(f"**Grid Size:** `{grid_size}` (preset locked)")
else:
    grid_size = st.sidebar.slider("Simulation Resolution", 20, 60, 40, 5)

log_fx = st.sidebar.slider("X Wave Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fx"], 0.1)
log_fy = st.sidebar.slider("Y Wave Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fy"], 0.1)
log_fz = st.sidebar.slider("Z Wave Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fz"], 0.1)

phase_x = np.radians(st.sidebar.slider("X Wave Phase (°)", 0, 360, preset["px"], 10))
phase_y = np.radians(st.sidebar.slider("Y Wave Phase (°)", 0, 360, preset["py"], 10))
phase_z = np.radians(st.sidebar.slider("Z Wave Phase (°)", 0, 360, preset["pz"], 10))

threshold = st.sidebar.slider("Pattern Intensity Threshold", 0.0, 1.0, preset["threshold"], 0.01)
lock_strength = st.sidebar.slider("Precision Window", 0.0, 1.0, preset["lock"], 0.005)

# Optional toggle
view_mode = st.sidebar.radio("Visualization Mode", ["Geometry Only", "Wave Overlay"])

fx, fy, fz = 10**log_fx, 10**log_fy, 10**log_fz
x = np.linspace(0, domain_scale, grid_size)
y = np.linspace(0, domain_scale, grid_size)
z = np.linspace(0, domain_scale, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

EX = np.sin(fx * np.pi * X + phase_x)
EY = np.sin(fy * np.pi * Y + phase_y)
EZ = np.sin(fz * np.pi * Z + phase_z)
interference = np.abs(EX * EY * EZ)

field_norm = (interference - interference.min()) / (interference.max() - interference.min())
lock_mask = ((field_norm > threshold - lock_strength) & (field_norm < threshold + lock_strength))

xv, yv, zv = X[lock_mask], Y[lock_mask], Z[lock_mask]
color_vals = field_norm[lock_mask]

if len(xv) > 0:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color=color_vals if view_mode == "Wave Overlay" else "white",
            colorscale='Viridis' if view_mode == "Wave Overlay" else None,
            opacity=0.6,
        )
    ))
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black', scene_bgcolor='black'
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("📷 Save Snapshot"):
        pio.write_image(fig, "TRR_snapshot.png", width=1000, height=800)
        st.success("Snapshot saved as TRR_snapshot.png")

else:
    st.warning("No visible geometry. Adjust intensity threshold or wave parameters.")

st.markdown(f"**Threshold**: {threshold:.2f} ± {lock_strength:.3f} — Frequencies: X=10^{log_fx:.1f}Hz, Y=10^{log_fy:.1f}Hz, Z=10^{log_fz:.1f}Hz")

# --- Collapsible TRR Explanation ---
with st.expander("📘 What Is TRR Isoplane Geometry?"):
    st.markdown("""
    **TRR (Theory of Rendered Reality)** models how reality appears only when wave-based fields align into coherent patterns.

    **Isoplane Geometry** happens when X, Y, and Z waves intersect *just right*—forming stable structures.

    **Simplified Equation:**
    > `resonance_energy > threshold`  
    > _When internal and external fields overlap enough, reality 'renders'._

    This viewer shows you **where** that happens in space.

    - High alignment = more structure
    - Phase mismatch = less pattern
    - Thresholds let you filter only the coherent zones
    """)
