import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def recommended_grid_size(frequencies_hz, domain_scale_m, target_grid=40, reference_freq_log10=6.0, reference_domain=1.0, min_pts=10, max_pts=100):
    c = 299_792_458  # m/s
    smallest_lambda = min(c / f for f in frequencies_hz if f > 0)
    cycles = domain_scale_m / smallest_lambda

    ref_freq = 10 ** reference_freq_log10
    ref_lambda = c / ref_freq
    ref_cycles = reference_domain / ref_lambda

    scaling_factor = cycles / ref_cycles
    grid_points = int(target_grid * scaling_factor)

    return max(min_pts, min(grid_points, max_pts))

def chladni_mode_to_waveparams(r: int, l: int, axis: str):
    """
    Maps Chladni plate mode numbers to frequency and phase values
    for TRR-style 3D interference simulation.

    Arguments:
    - r: radial mode (number of circular nodes)
    - l: angular mode (number of nodal diameters)
    - axis: 'x', 'y', or 'z'

    Returns:
    - frequency (log10 Hz), phase (degrees)
    """

    base_log_freq = 6.0  # Typical EM vibration base (~1 MHz)
    axis_shift = {'x': 0.0, 'y': 0.1, 'z': 0.2}[axis]  # Slight shifts to avoid perfect overlap

    freq = base_log_freq + 0.3 * r + axis_shift
    phase = (l * 90) % 360

    return freq, phase

# --- Preset Extension for Chladni Modes ---
fx1, px1 = chladni_mode_to_waveparams(2, 3, 'x')
fy1, py1 = chladni_mode_to_waveparams(2, 3, 'y')
fz1, pz1 = chladni_mode_to_waveparams(2, 3, 'z')

fx2, px2 = chladni_mode_to_waveparams(1, 2, 'x')
fy2, py2 = chladni_mode_to_waveparams(1, 2, 'y')
fz2, pz2 = chladni_mode_to_waveparams(2, 3, 'z')



#presets.update(chladni_presets)

st.set_page_config(layout="wide")
st.title("Theory of Rendered Reality Isoplane Geometry Simulator V10")

st.markdown("""
This simulator shows how resonance waves in **X, Y, and Z** create combined **cymatic patterns**—stable 3D geometries that emerge when the waves intersect.
""")
with st.expander("What Are Cymatic Patterns in TRR?"):
    st.markdown("""
    **Cymatic patterns** are geometric shapes that emerge when **standing waves interact**—typically visualized in 2D with sand on vibrating plates.

    In **TRR**, we simulate **3D cymatics** using **resonant wave interference across X, Y, and Z axes**.

    These structures represent **rendered zones**—areas where internal and external fields overlap in a way that satisfies the rendering condition:
    
    > **|Ψᵣ · Φ|² > Tᵣ**

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
        "threshold": 0.05, "lock": 0.01,
        "grid_size": 40,
        "domain_scale": 1
    },
    "Cymatic Shell": {
        "desc": "Layered wave harmonics generate cymatic-like structures. This preset mimics sound-driven geometry—where resonance creates shells of stillness.",
        "fx": 3.0, "fy": 4.0, "fz": 4.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.1, "lock": 0.02,
        "grid_size": 40,
        "domain_scale": 22
    },
    "Render Fog": {
        "desc": "Rendering is a struggle in this chaotic field. Fields are almost coherent, but never quite stabilize—like trying to see through shifting mist.",
        "fx": 5.5, "fy": 6.0, "fz": 6.5,
        "px": 90, "py": 45, "pz": 180,
        "threshold": 0.3, "lock": 0.09,
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
        "threshold": 0.15, "lock": 0.06,
        "grid_size": 40,
        "domain_scale": 17.0
    },
    "Singularity Shell": {
        "desc": "A black hole–inspired collapse field. All rendering is pushed to the outer fringe — the center is a void where nothing can render.",
        "fx": -1.0, "fy": -1.0, "fz": 17.0,
        "px": 90, "py": 110, "pz": 40,
        "threshold": 0.04, "lock": 0.01,
        "grid_size": 60,
        "domain_scale": 18.0
    },
    "Quantum Limit": {
        "desc": "This pattern sits just below the rendering threshold. Resonance threads emerge and vanish — a delicate dance at the edge of realization.",
        "fx": 5.1, "fy": 4.9, "fz": 5.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.01, "lock": 0.002,
        "grid_size": 40,
        "domain_scale": 1.0
    },
    "Solar Lattice": {
        "desc": "Photonic field interaction in the solar coherence band — modeling solar perception windows.",
        "fx": 14.8, "fy": 15.0, "fz": 15.2,
        "px": 0, "py": 45, "pz": 90,
        "threshold": 0.05, "lock": 0.01,
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
        "threshold": 0.02, "lock": 0.004,
        "domain_scale": 11.0,
        "grid_size": 30
    },
    "Toroidal Helix (r=2, l=3)": {
    "fx": 6.6, "fy": 6.7, "fz": 6.8,
    "px": 270, "py": 0, "pz": 270,
    "threshold": 0.5,
    "lock": 0.01,
    "grid_size": 75,
    "domain_scale": 6.0,
    "desc": "Clear toroidal Chladni resonance with full symmetry — ideal for visualizing harmonic confinement in TRR."
},

"Axial Helix (r=1, l=2)": {
    "fx": 6.3, "fy": 6.4, "fz": 6.6,
    "px": 180, "py": 0, "pz": 180,
    "threshold": 0.95,
    "lock": 0.01,
    "grid_size": 75,
    "domain_scale": 1.0,
    "desc": "Axial harmonic resonance based on fundamental Chladni symmetry — mimics bio-coherent spine-wave interaction."
}

}

selected = st.sidebar.selectbox("**TRR Demo Presets:**", list(presets.keys()))
preset = presets[selected]


st.sidebar.markdown(f"**Description:** {preset['desc']}")

if "last_preset" not in st.session_state or st.session_state.last_preset != selected:
    st.session_state.last_preset = selected

    # Frequencies
    st.session_state.log_fx = preset["fx"]
    st.session_state.log_fy = preset["fy"]
    st.session_state.log_fz = preset["fz"]

    # Grid
    st.session_state.grid_size = preset.get("grid_size", 40)

    # Threshold and lock
    st.session_state.threshold = preset.get("threshold", 0.05)
    st.session_state.lock_strength = preset.get("lock", 0.01)

    # Phase
    st.session_state.phase_x = preset.get("px", 0)
    st.session_state.phase_y = preset.get("py", 0)
    st.session_state.phase_z = preset.get("pz", 0)

    
# --- Recommended Settings Helper ---
helper_ranges = {
    "Resonant Core (Ψₐ ∩ Φₐ)":       {"grid": "40",    "domain": "1–3"},
    "Phase Rift":                    {"grid": "40",    "domain": "2–4"},
    "Cymatic Shell":                 {"grid": "40–50", "domain": "20–25"},
    "Render Fog":                    {"grid": "35–40", "domain": "5–8"},
    "Perceptual Jam":                {"grid": "50–60", "domain": "25–30"},
    "Biofield Bloom":                {"grid": "40–50", "domain": "15–20"},
    "Singularity Shell":             {"grid": "60",    "domain": "18–20"},
    "Quantum Limit":                 {"grid": "40",    "domain": "1–3"},
    "Solar Lattice":                 {"grid": "40",    "domain": "4–6"},
    "Quartz Memory Shell":           {"grid": "50",    "domain": "1–3"},
    "Schumann Phase Gate":           {"grid": "30",    "domain": "9–11"},
    "Toroidal Helix (r=2, l=3)":      {"grid": "55",    "domain": "1.0"},
    "Axial Helix (r=1, l=2)":         {"grid": "75",    "domain": "1.0"},
}


st.sidebar.markdown("Recommended Settings")
recommend = helper_ranges.get(selected, {})
st.sidebar.markdown(f"**Domain Size:** {recommend.get('domain', '—')}")
st.sidebar.markdown(f"**Grid Resolution:** {recommend.get('grid', '—')}")
st.sidebar.markdown("---")

domain_scale_default = float(preset.get("domain_scale", 10.0))
domain_scale = st.sidebar.slider("Display Domain Size", 1.0, 30.0, float(preset.get("domain_scale", 10.0)), step=1.0)

st.sidebar.markdown("---")

if "last_preset" not in st.session_state or st.session_state.last_preset != selected:
    st.session_state.log_fx = preset["fx"]
    st.session_state.log_fy = preset["fy"]
    st.session_state.log_fz = preset["fz"]
    st.session_state.last_preset = selected

log_fx = st.sidebar.slider("X Wave Frequency (log₁₀ Hz)", -1.0, 17.0, value=st.session_state.log_fx, step=0.1, key="log_fx")
log_fy = st.sidebar.slider("Y Wave Frequency (log₁₀ Hz)", -1.0, 17.0, value=st.session_state.log_fy, step=0.1, key="log_fy")
log_fz = st.sidebar.slider("Z Wave Frequency (log₁₀ Hz)", -1.0, 17.0, value=st.session_state.log_fz, step=0.1, key="log_fz")


fx = 10**log_fx
fy = 10**log_fy
fz = 10**log_fz

if "grid_size" not in st.session_state or st.session_state.get("last_preset") != selected:
    st.session_state.grid_size = preset.get("grid_size", 40)
    st.session_state.last_preset = selected 

grid_size = st.sidebar.slider("Geometry Detail (Grid Resolution)", 20, 100, value=st.session_state.grid_size, step=5, key="grid_size")
# Force grid size to be odd for symmetric centering
if grid_size % 2 == 0:
    grid_size += 1
    
st.sidebar.markdown("---")

threshold = st.sidebar.slider("Render Threshold", 0.0, 1.0, value=st.session_state.get("threshold", 0.05), step=0.01, key="threshold")
lock_strength = st.sidebar.slider("Resonance Lock Range", 0.0, 1.0, value=st.session_state.lock_strength, step=0.005, key="lock_strength")


st.sidebar.markdown("---")
st.sidebar.markdown("""
**Phase Shift Help**
- Affects the **starting point** of the wave
- 0° = Perfect Alignment
- 180° = Destructive Interference
- 90° / 270° = Orthogonal Field States, often incoherent
""")
st.sidebar.markdown("Wave Phase Settings (Degrees)")
phase_x = np.radians(st.sidebar.slider("X-Axis Phase (°)", 0, 360, value=st.session_state.phase_x, step=10, key="phase_x"))
phase_y = np.radians(st.sidebar.slider("Y-Axis Phase (°)", 0, 360, value=st.session_state.phase_y, step=10, key="phase_y"))
phase_z = np.radians(st.sidebar.slider("Z-Axis Phase (°)", 0, 360, value=st.session_state.phase_z, step=10, key="phase_z"))

st.sidebar.markdown("---")

# Optional toggle
view_mode = st.sidebar.radio("Visualization Mode", ["Geometry Only", "Wave Overlay"], index=1)

use_chladni = st.sidebar.checkbox("Enable Chladni Mode Input")

if use_chladni:
    use_chladni_override = st.sidebar.checkbox("Override Chladni Freq/Phase", value=False)

    st.sidebar.markdown("### Chladni Mode Selection")
    r_x = st.sidebar.slider("Radial Mode rₓ", 0, 4, 2)
    l_x = st.sidebar.slider("Angular Mode lₓ", 0, 4, 1)
    r_y = st.sidebar.slider("Radial Mode rᵧ", 0, 4, 2)
    l_y = st.sidebar.slider("Angular Mode lᵧ", 0, 4, 2)
    r_z = st.sidebar.slider("Radial Mode r𝓏", 0, 4, 2)
    l_z = st.sidebar.slider("Angular Mode l𝓏", 0, 4, 3)

    x = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
    y = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
    z = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    EX = np.sin(fx * np.pi * X + phase_x)
    EY = np.sin(fy * np.pi * Y + phase_y)
    EZ = np.sin(fz * np.pi * Z + phase_z)
    interference = np.abs(EX * EY * EZ)

    field_norm = (interference - interference.min()) / (interference.max() - interference.min())
    lock_mask = ((field_norm > threshold - lock_strength) & (field_norm < threshold + lock_strength))

    xv, yv, zv = X[lock_mask], Y[lock_mask], Z[lock_mask]
    color_vals = field_norm[lock_mask]


    if not use_chladni_override:
        log_fx, phase_x_deg = chladni_mode_to_waveparams(r_x, l_x, 'x')
        log_fy, phase_y_deg = chladni_mode_to_waveparams(r_y, l_y, 'y')
        log_fz, phase_z_deg = chladni_mode_to_waveparams(r_z, l_z, 'z')

        fx = 10 ** log_fx
        fy = 10 ** log_fy
        fz = 10 ** log_fz

        phase_x = np.radians(phase_x_deg)
        phase_y = np.radians(phase_y_deg)
        phase_z = np.radians(phase_z_deg)

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


    st.sidebar.markdown(f"**Chladni Mode Summary:**")
    st.sidebar.markdown(f"- X: r={r_x}, l={l_x}")
    st.sidebar.markdown(f"- Y: r={r_y}, l={l_y}")
    st.sidebar.markdown(f"- Z: r={r_z}, l={l_z}")


with st.sidebar:
    st.markdown("## Resources")
    with open("Rendered_Reality_TimG.pdf", "rb") as f:
        st.download_button(
            label="Download TRR White Paper",
            data=f,
            file_name="Rendered_Reality_TimG.pdf",
            mime="application/pdf"
        )






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

def format_hz(value):
    if value < 1e3:
        return f"{value:.2f} Hz"
    elif value < 1e6:
        return f"{value/1e3:.2f} kHz"
    elif value < 1e9:
        return f"{value/1e6:.2f} MHz"
    elif value < 1e12:
        return f"{value/1e9:.2f} GHz"
    elif value < 1e15:
        return f"{value/1e12:.2f} THz"
    else:
        return f"{value/1e15:.2f} PHz"

fx, fy, fz = 10**log_fx, 10**log_fy, 10**log_fz

def format_wavelength(frequency_hz):
    c = 299_792_458  # Speed of light in m/s
    if frequency_hz == 0:
        return "∞"
    wavelength_m = c / frequency_hz
    if wavelength_m >= 1:
        return f"{wavelength_m:.2f} m"
    elif wavelength_m >= 1e-3:
        return f"{wavelength_m * 1e3:.2f} mm"
    elif wavelength_m >= 1e-6:
        return f"{wavelength_m * 1e6:.2f} µm"
    elif wavelength_m >= 1e-9:
        return f"{wavelength_m * 1e9:.2f} nm"
    else:
        return f"{wavelength_m:.2e} m"
        
st.markdown(f"""
**Render Threshold:** {threshold:.2f}  
**Resonance Lock Range:** ±{lock_strength:.3f}  
**Wave Frequencies and Wavelengths:**  
- X Axis: {format_hz(fx)}  (λ = {format_wavelength(fx)})  
- Y Axis: {format_hz(fy)}  (λ = {format_wavelength(fy)})  
- Z Axis: {format_hz(fz)}  (λ = {format_wavelength(fz)})
""")


st.markdown("""
---
**TRR Equation of Rendered Geometry**

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
