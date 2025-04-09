import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

st.set_page_config(layout="wide")
st.title("Theory of Rendered Reality Isoplane Geometry Simulator V10")

st.markdown("""
This simulator shows how resonance waves in **X, Y, and Z** create combined **cymatic patterns**—stable 3D geometries that emerge when resonance waves intersect.
""")
with st.expander("🔊 What Are Cymatic Patterns in TRR?"):
    st.markdown("""
    **Cymatic patterns** are geometric shapes that emerge when **standing waves interact**—typically visualized in 2D with sand on vibrating plates.

    In **TRR**, we simulate **3D cymatics** using **resonant wave interference across X, Y, and Z axes**.

    These structures represent **rendered zones**—areas where internal and external fields overlap in a way that satisfies the rendering condition:
    
    > `|Ψᵣ · Φ|² > Tᵣ`

    - Nodes (no movement) = voids in perception  
    - Antinodes (intense overlap) = structured reality

    Think of this as a **3D “Chladni field”** where **reality crystallizes** into visible form. Different presets simulate:
    
    - Phase alignment vs. decoherence
    - Biological vs. cosmological rendering
    - Perceptual jamming vs. structured emergence
    """)


# --- Preset Definitions ---
presets = {
    "Resonant Core (Ψₐ ∩ Φₐ)": {
        "desc": "A perfect match of internal and external waveforms—reality crystallizes at the center. This is the golden zone where rendering is guaranteed.",
        "fx": 6.0, "fy": 6.0, "fz": 6.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.05, "lock": 0.03,
        "grid_size": 40,
        "domain_scale": 1
    },
    "Phase Rift": {
        "desc": "One axis breaks coherence—TRR shows how a single misalignment disrupts what is rendered. Like trying to tune a radio with one knob off.",
        "fx": 6.0, "fy": 6.0, "fz": 6.0,
        "px": 45, "py": 0, "pz": 0,
        "threshold": 0.05, "lock": 0.03,
        "grid_size": 40,
        "domain_scale": 1
    },
    "Cymatic Shell": {
        "desc": "Layered wave harmonics generate cymatic-like structures. This preset mimics sound-driven geometry—where resonance creates shells of stillness.",
        "fx": 3.0, "fy": 4.0, "fz": 4.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.1, "lock": 0.05,
        "grid_size": 40,
        "domain_scale": 22
    },
    "Render Fog": {
        "desc": "Rendering is a struggle in this chaotic field. Fields are almost coherent, but never quite stabilize—like trying to see through shifting mist.",
        "fx": 5.5, "fy": 6.0, "fz": 6.5,
        "px": 90, "py": 45, "pz": 180,
        "threshold": 0.3, "lock": 0.15,
        "grid_size": 40,
        "domain_scale": 3
    },
    "Perceptual Jam": {
        "desc": "When your perception filter is 90° out of sync, nothing gets through. The system denies rendering—reality blinks out.",
        "fx": 7.0, "fy": 7.0, "fz": 7.0,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.05, "lock": 0.01,
        "grid_size": 60,
        "domain_scale": 30
    },
    "Biofield Bloom": {
        "desc": "A biological coherence pattern — local regions resonate while the surrounding field remains inert. Similar to consciousness arising in neurons.",
        "fx": 2.5, "fy": 3.0, "fz": 2.8,
        "px": 50, "py": 0, "pz": 150,
        "threshold": 0.15, "lock": 0.10,
        "grid_size": 40,
        "domain_scale": 17.0
    },
    "Singularity Shell": {
        "desc": "A black hole–inspired collapse field. All rendering is pushed to the outer fringe — the center is a void where nothing can render.",
        "fx": -1.0, "fy": -1.0, "fz": 17.0,
        "px": 90, "py": 110, "pz": 40,
        "threshold": 0.04, "lock": 0.02,
        "grid_size": 60,
        "domain_scale": 18.0
    },
    "Quantum Limit": {
        "desc": "This pattern sits just below the rendering threshold. Resonance threads emerge and vanish — a delicate dance at the edge of realization.",
        "fx": 5.1, "fy": 4.9, "fz": 5.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.01, "lock": 0.01,
        "grid_size": 40,
        "domain_scale": 1.0
    },
    "Solar Lattice": {
        "desc": "Photonic field interaction in the solar coherence band — modeling solar perception windows.",
        "fx": 14.8, "fy": 15.0, "fz": 15.2,
        "px": 0, "py": 45, "pz": 90,
        "threshold": 0.05, "lock": 0.025,
        "domain_scale": 5.0,
        "grid_size": 40
    },
    "Quartz Memory Shell": {
        "desc": "Highly stable crystalline resonance — visualizing field anchoring inside silica-based substrates.",
        "fx": 12.1, "fy": 12.1, "fz": 12.1,
        "px": 0, "py": 120, "pz": 240,
        "threshold": 0.08, "lock": 0.02,
        "domain_scale": 3.0,
        "grid_size": 50
    },
    "Schumann Phase Gate": {
        "desc": "Earth’s electromagnetic coherence frequency — rendering resonance aligned with planetary rhythms.",
        "fx": 0.89, "fy": 0.91, "fz": 0.93,
        "px": 0, "py": 90, "pz": 180,
        "threshold": 0.02, "lock": 0.01,
        "domain_scale": 11.0,
        "grid_size": 30
    }
}

selected = st.sidebar.selectbox("Choose TRR Demo Preset", list(presets.keys()))
preset = presets[selected]

st.sidebar.markdown(f"**Description:** {preset['desc']}")

# --- Recommended Settings Helper ---
helper_ranges = {
    "Resonant Core (Ψₐ ∩ Φₐ)":       {"grid": "30–40", "domain": "1–3"},
    "Phase Rift":                    {"grid": "40",    "domain": "2–4"},
    "Cymatic Shell":                 {"grid": "40–50", "domain": "20–25"},
    "Render Fog":                    {"grid": "35–40", "domain": "5–8"},
    "Perceptual Jam":                {"grid": "50–60", "domain": "25–30"},
    "Biofield Bloom":                {"grid": "40–50", "domain": "15–20"},
    "Singularity Shell":            {"grid": "60",    "domain": "18–20"},
    "Quantum Limit":                {"grid": "40",    "domain": "1–3"},
    "Solar Lattice":                {"grid": "40",    "domain": "4–6"},
    "Quartz Memory Shell":          {"grid": "50",    "domain": "1–3"},
    "Schumann Phase Gate":          {"grid": "30",    "domain": "9–11"},
}

st.sidebar.markdown("Recommended Settings")
recommend = helper_ranges.get(selected, {})
st.sidebar.markdown(f"**Domain Size:** {recommend.get('domain', '—')}")
st.sidebar.markdown(f"**Grid Resolution:** {recommend.get('grid', '—')}")
st.sidebar.markdown("---")

domain_scale_default = float(preset.get("domain_scale", 10.0))
domain_scale = st.sidebar.slider(
    "Display Domain Size",
    min_value=1.0,
    max_value=30.0,
    value=domain_scale_default,
    step=1.0
)

grid_size = st.sidebar.slider("Geometry Detail (Grid Resolution)", 20, 60, 40, 5)

st.sidebar.markdown("---")

log_fx = st.sidebar.slider("X Wave Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fx"], 0.1)
log_fy = st.sidebar.slider("Y Wave Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fy"], 0.1)
log_fz = st.sidebar.slider("Z Wave Frequency (log₁₀ Hz)", -1.0, 17.0, preset["fz"], 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Phase Shift Help**
- Affects the **starting point** of the wave
- 0° = Perfect Alignment
- 180° = Destructive Interference
- 90° / 270° = Orthogonal Field States, often incoherent
""")
st.sidebar.markdown("Wave Phase Settings (Degrees)")
phase_x = np.radians(st.sidebar.slider("X-Axis Phase (°)", 0, 360, preset["px"], 10))
phase_y = np.radians(st.sidebar.slider("Y-Axis Phase (°)", 0, 360, preset["py"], 10))
phase_z = np.radians(st.sidebar.slider("Z-Axis Phase (°)", 0, 360, preset["pz"], 10))


st.sidebar.markdown("---")

threshold = st.sidebar.slider("Render Threshold", 0.0, 1.0, preset["threshold"], 0.01)
lock_strength = st.sidebar.slider("Resonance Lock Range", 0.0, 1.0, preset["lock"], 0.005)

st.sidebar.markdown("---")

# Optional toggle
view_mode = st.sidebar.radio("Visualization Mode", ["Geometry Only", "Wave Overlay"], index=1)

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
        paper_bgcolor='black', scene_bgcolor='black',
        height=640
    )
    st.plotly_chart(fig, use_container_width=True)

    
else:
    st.warning("No visible geometry. Adjust intensity threshold or wave parameters.")

st.markdown(f"**Threshold**: {threshold:.2f} ± {lock_strength:.3f} — Frequencies: X=10^{log_fx:.1f}Hz, Y=10^{log_fy:.1f}Hz, Z=10^{log_fz:.1f}Hz")
st.markdown("""
---
### 📐 TRR Equation of Rendered Geometry

> **Render Condition:**  
> \\[ |⟨ \\Psi_r(x, t) | H_{res} | \\Phi(x, t) ⟩|^2 > T_r \\]

This simulator visualizes the points in 3D space where this equation **crosses the render threshold**—where reality takes form.
""")

# --- Collapsible TRR Explanation ---
with st.expander("📘 What Is TRR Isoplane Geometry?"):
    st.markdown("""
    **TRR (Theory of Rendered Reality)** models how reality appears when wave-based resonance fields align into coherent patterns.

    **Isoplane Geometry** happens when X, Y, and Z waves intersect *just right*—forming stable structures.

    **Simplified Equation:**
    > `resonance_energy > threshold`  
    > _When internal and external fields overlap enough, reality 'renders'._

    This viewer shows you **where** that happens in space.

    - High alignment = more structure
    - Phase mismatch = less pattern
    - Thresholds let you filter only the coherent zones
    """)
