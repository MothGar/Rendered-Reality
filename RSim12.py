import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import math

def clamp_log(value, minval=-3.0, maxval=20.0):
    # If value is missing, non-numeric, or nan, default to 6.0
    try:
        val = float(value)
        if math.isnan(val):
            val = 6.0
    except:
        val = 6.0
    return max(min(val, maxval), minval)
    
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

def suggest_domain_scale(frequencies_hz, cycles=3):
    """
    Suggest domain size to fit ~'cycles' full wave cycles of the smallest frequency component.
    """
    c = 299_792_458  # Speed of light in m/s
    wavelengths = [c / f if f > 0 else 1.0 for f in frequencies_hz]
    shortest_wavelength = min(wavelengths)
    return cycles * shortest_wavelength


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
        "chladni": False,
        "fx": 6.0, "fy": 6.0, "fz": 6.0,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.05, "lock": 0.03,
        "grid_size": 55,
        "domain_scale": 1.3
    },
    "Phase Rift": {
        "desc": "One axis breaks coherence—TRR shows how a single misalignment disrupts what is rendered. Like trying to tune a radio with one knob off.",
        "chladni": False,
        "fx": 6.0, "fy": 6.0, "fz": 6.0,
        "px": 45, "py": 90, "pz": 90,
        "threshold": 0.05, "lock": 0.01,
        "grid_size": 75,
        "domain_scale": 5.4
    },
    "Cymatic Shell": {
        "desc": "Layered wave harmonics generate cymatic-like structures. This preset mimics sound-driven geometry—where resonance creates shells of stillness.",
        "chladni": False,
        "fx": 3.0, "fy": 4.0, "fz": 4.0,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.1, "lock": 0.02,
        "grid_size": 75,
        "domain_scale": 1
    },
    "Render Fog": {
        "desc": "Rendering is a struggle in this chaotic field. Fields are almost coherent, but never quite stabilize—like trying to see through shifting mist.",
        "chladni": False,
        "fx": 5.5, "fy": 6.0, "fz": 6.5,
        "px": 90, "py": 45, "pz": 180,
        "threshold": 0.3, "lock": 0.09,
        "grid_size": 45,
        "domain_scale": 3
    },
    "Perceptual Jam": {
        "desc": "When your perception filter is 90° out of sync, nothing gets through. The system denies rendering—reality blinks out.",
        "chladni": False,
        "fx": 7.0, "fy": 7.0, "fz": 7.0,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.05, "lock": 0.01,
        "grid_size": 85,
        "domain_scale": 30
    },
    "Biofield Bloom": {
        "desc": "**Activate Chladni Mode:** Discrete coherence zones emerge in layers — like neural activation patterns surrounded by inert space. Represents localized awareness within a passive field.",
        "chladni": True,
        "fx": 13.9, "fy": 13.7, "fz": 13.5,
        "px": 50, "py": 0, "pz": 180,
        "threshold": 0.15,
        "lock": 0.06,
        "grid_size": 75,
        "domain_scale": 2.0,
        "r_x": 2, "l_x": 2,
        "r_y": 2, "l_y": 0,
        "r_z": 1, "l_z": 3
    },
    "Singularity Shell": {
        "desc": "**Activate Chladni Mode:** A black hole–inspired collapse field. All rendering is pushed to the outer fringe — the center is a void where nothing can render.",
        "chladni": True,
        "fx": 6, "fy": 6, "fz": 9,
        "px": 90, "py": 180, "pz": 0,
        "threshold": 0.22,
        "lock": 0.06,
        "grid_size": 50,
        "domain_scale": 9.2,
        "r_x": 0, "l_x": 0,
        "r_y": 0, "l_y": 0,
        "r_z": 4, "l_z": 3
    },
    "Quantum Limit": {
        "desc": "**Activate Chladni Mode:** This pattern sits just below the rendering threshold. Resonance threads emerge and vanish — a delicate dance at the edge of realization.",
        "chladni": True,
        "fx": 5.1, "fy": 4.9, "fz": 5.0,
        "px": 0, "py": 0, "pz": 0,
        "threshold": 0.01, "lock": 0.002,
        "grid_size": 40,
        "domain_scale": 1.0
    },
    "Solar Lattice": {
        "desc": "**Activate Chladni Mode:** High-energy photonic resonance net — simulates structured coherence zones in solar EM emission windows (UV-visible band). Useful for visualizing spectral emission coherence during photosphere-level observation.",
        "chladni": True,
        "fx": 14.7, "fy": 14.9, "fz": 15.1,
        "px": 45, "py": 90, "pz": 135,
        "threshold": 0.06,
        "lock": 0.015,
        "domain_scale": 6.0,
        "grid_size": 65,
        "r_x": 1, "l_x": 2,
        "r_y": 2, "l_y": 1,
        "r_z": 3, "l_z": 2
    },
    "Quartz Memory Shell": {
        "desc": "**Activate Chladni Mode:** Encoded memory band with coherence column and null-protected axis. Preserves state integrity with phase stratification.",
        "chladni": True,
        "fx": 4.8, "fy": 4.2, "fz": 5.4,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.11,
        "lock": 0.014,
        "grid_size": 65,
        "domain_scale": 2.4,
        "r_x": 1, "l_x": 1,
        "r_y": 0, "l_y": 1,
        "r_z": 2, "l_z": 3
    },
    "Schumann Phase Gate": {
        "desc": "Earth’s electromagnetic coherence frequency — rendering resonance aligned with planetary rhythms.",
        "chladni": False,
        "fx": 0.89, "fy": 0.91, "fz": 0.93,
        "px": 0, "py": 90, "pz": 180,
        "threshold": 0.02, "lock": 0.004,
        "domain_scale": 11.0,
        "grid_size": 30
    },
    "Toroidal Helix (r=2, l=3)": {
        "desc": "**Activate Chladni Mode:** Generates a toroidal standing-wave field with axial bypass symmetry. Triple angular nodes wrap in X and Z, while Y maintains radial structure with no angular twist. Useful for modeling vertical energy channels through toroidal confinement zones—such as magnetic bottle cores or hypothetical consciousness conduits.",
        "chladni": True,
        "fx": 6.6, "fy": 6.7, "fz": 6.8,
        "px": 270, "py": 0, "pz": 270,
        "threshold": 0.5,
        "lock": 0.01,
        "grid_size": 75,
        "domain_scale": 2.0,
        "r_x": 2, "l_x": 3,
        "r_y": 2, "l_y": 0,
        "r_z": 2, "l_z": 3
    },
    "Axial Helix (r=1, l=2)": {
        "chladni": True,
        "fx": 6.3, "fy": 6.4, "fz": 6.6,
        "px": 180, "py": 0, "pz": 180,
        "threshold": 0.95,
        "lock": 0.01,
        "grid_size": 75,
        "domain_scale": 1.0,
        "r_x": 2, "l_x": 2,
        "r_y": 1, "l_y": 2,
        "r_z": 1, "l_z": 1,
        "desc": "**Activate Chladni Mode:** Axial harmonic resonance with a lateral twist. X and Y axes spiral asymmetrically into a central spine as Z maintains a base-mode torsion — reminiscent of neurological phase entrainment."
},
    "Toroidal Core": {
        "desc": "**Activate Chladni Mode:** Dense toroidal symmetry with full inner cohesion",
        "chladni": True,
        "fx": 13.5, "fy": 13.5, "fz": 13.5,
        "px": 0, "py": 120, "pz": 240,
        "threshold": 0.05, "lock": 0.04,
        "grid_size": 75, "domain_scale": 2.0,
        "r_x": 2, "l_x": 1,
        "r_y": 2, "l_y": 2,
        "r_z": 2, "l_z": 3
},
    "Cellular Cubes": {
        "desc": "**Activate Chladni Mode:** Nested cubic interference pattern — external symmetry encloses null interiors. Mimics quantum-tessellation failure zones.",
        "chladni": True,
        "fx": 13.7, "fy": 13.7, "fz": 13.7,
        "px": 90, "py": 120, "pz": 90,
        "threshold": 0.03, "lock": 0.01,
        "grid_size": 75, "domain_scale": 2.9,
        "r_x": 1, "l_x": 1,
        "r_y": 1, "l_y": 2,
        "r_z": 1, "l_z": 1
},
    "Phase Tuned Cross Lattice": {
        "desc": "Resonant interference throughout grid, high coherence crosswave",
        "chladni": False,
        "fx": 13.5, "fy": 13.5, "fz": 13.5,
        "px": 90, "py": 90, "pz": 90,
        "threshold": 0.05, "lock": 0.04,
        "grid_size": 70, "domain_scale": 1.5,
        
},
    "Phase Collapse Gate": {
        "desc": "**Activate Chladni Mode:** Flat stacked planar emergence at destructive Z",
        "chladni": True,
        "fx": 13.7, "fy": 13.7, "fz": 13.7,
        "px": 90, "py": 120, "pz": 180,
        "threshold": 0.05, "lock": 0.04,
        "grid_size": 50, "domain_scale": 1.0,
        "r_x": 1, "l_x": 1,
        "r_y": 1, "l_y": 2,
        "r_z": 1, "l_z": 0
},
    "Ring Cross": {
        "desc": "**Activate Chladni Mode:** Circular ring with orthogonal cut planes",
        "chladni": True,
        "fx": 13.7, "fy": 13.7, "fz": 13.7,
        "px": 90, "py": 120, "pz": 90,
        "threshold": 0.05, "lock": 0.03,
        "grid_size": 55, "domain_scale": 1.5,
        "r_x": 1, "l_x": 1,
        "r_y": 1, "l_y": 2,
        "r_z": 1, "l_z": 0
}
    

}

selected = st.sidebar.selectbox("**TRR Demo Presets:**", list(presets.keys()))
preset = presets[selected]
# Automatically activate Chladni mode if the preset requires it
if "chladni" in preset:
    st.session_state.use_chladni = preset["chladni"]


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

    # Chladni Modes — if available
    st.session_state.r_x = preset.get("r_x", 2)
    st.session_state.l_x = preset.get("l_x", 2)
    st.session_state.r_y = preset.get("r_y", 2)
    st.session_state.l_y = preset.get("l_y", 2)
    st.session_state.r_z = preset.get("r_z", 2)
    st.session_state.l_z = preset.get("l_z", 2)


    
# --- Recommended Settings Helper ---
helper_ranges = {
    "Resonant Core (Ψₐ ∩ Φₐ)":       {"grid": "55",    "domain": "1.3"},
    "Phase Rift":                    {"grid": "75",    "domain": "5.4"},
    "Cymatic Shell":                 {"grid": "75",    "domain": "1"},
    "Render Fog":                    {"grid": "45",    "domain": "3"},
    "Perceptual Jam":                {"grid": "85",    "domain": "30"},
    "Biofield Bloom":                {"grid": "75",    "domain": "2.0"},
    "Singularity Shell":             {"grid": "50",    "domain": "9.2"},
    "Quantum Limit":                 {"grid": "40",    "domain": "1.0"},
    "Solar Lattice":                 {"grid": "65",    "domain": "6.0"},
    "Quartz Memory Shell":           {"grid": "65",    "domain": "2.4"},
    "Schumann Phase Gate":           {"grid": "30",    "domain": "11.0"},
    "Toroidal Helix (r=2, l=3)":      {"grid": "75",    "domain": "2.0"},
    "Axial Helix (r=1, l=2)":         {"grid": "75",    "domain": "1.0"},
    "Toroidal Core":                 {"grid": "75",    "domain": "2.0"},
    "Cellular Cubes":                {"grid": "75",    "domain": "2.9"},
    "Phase Tuned Cross Lattice":     {"grid": "70",    "domain": "1.5"},
    "Phase Collapse Gate":           {"grid": "50",    "domain": "1.0"},
    "Ring Cross":                    {"grid": "55",    "domain": "1.5"}
}


st.sidebar.markdown("Recommended Settings")
recommend = helper_ranges.get(selected, {})
st.sidebar.markdown(f"**Domain Size:** {recommend.get('domain', '—')}")
st.sidebar.markdown(f"**Grid Resolution:** {recommend.get('grid', '—')}")
st.sidebar.markdown("---")

# Fallback-safe log values
log_fx_val = clamp_log(st.session_state.get("log_fx", 6.0))
log_fy_val = clamp_log(st.session_state.get("log_fy", 6.0))
log_fz_val = clamp_log(st.session_state.get("log_fz", 6.0))

# Now create sliders safely
log_fx = st.sidebar.slider("X Wave Frequency (log₁₀ Hz)", -3.0, 20.0, value=log_fx_val, step=0.1, key="log_fx")
log_fy = st.sidebar.slider("Y Wave Frequency (log₁₀ Hz)", -3.0, 20.0, value=log_fy_val, step=0.1, key="log_fy")
log_fz = st.sidebar.slider("Z Wave Frequency (log₁₀ Hz)", -3.0, 20.0, value=log_fz_val, step=0.1, key="log_fz")

# Convert frequency sliders to linear Hz
fx = 10**log_fx
fy = 10**log_fy
fz = 10**log_fz

# Compute shortest wavelength and suggest domain
try:
    shortest_lambda = min([fx, fy, fz])
    auto_domain = suggest_domain_scale([fx, fy, fz], cycles=3)

    st.sidebar.markdown("**Wave Cycle-Based Domain Suggestion:**")
    st.sidebar.markdown(f"- Shortest λ = {format_wavelength(shortest_lambda)}")
    st.sidebar.markdown(f"- Suggested Domain ≈ {auto_domain:.3f} m  (for ~3 cycles)")
except Exception as e:
    auto_domain = 1.0
    st.sidebar.markdown("**Wave Cycle-Based Domain Suggestion:**")
    st.sidebar.markdown("- Error calculating wavelength.")
    st.sidebar.caption(str(e))

# Use either preset domain or auto-suggested one
domain_scale = st.sidebar.slider(
    "Display Domain Size", 0.01, 30.0,
    float(preset.get("domain_scale", auto_domain)),
    step=0.1
)

st.sidebar.markdown("---")

    
if "last_preset" not in st.session_state or st.session_state.last_preset != selected:
    st.session_state.log_fx = preset["fx"]
    st.session_state.log_fy = preset["fy"]
    st.session_state.log_fz = preset["fz"]
    st.session_state.last_preset = selected





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

use_chladni = st.sidebar.checkbox("Enable Chladni Mode Input", value=st.session_state.get("use_chladni", False))

# Define override and sliders immediately if Chladni mode is on
if use_chladni:
    use_chladni_override = st.sidebar.checkbox("Override Chladni Freq/Phase", value=False)

    st.sidebar.markdown("### Chladni Mode Selection")
    r_x = st.sidebar.slider("Radial Mode rₓ", 0, 4, st.session_state.get("r_x", 2))
    l_x = st.sidebar.slider("Angular Mode lₓ", 0, 4, st.session_state.get("l_x", 2))
    r_y = st.sidebar.slider("Radial Mode rᵧ", 0, 4, st.session_state.get("r_y", 2))
    l_y = st.sidebar.slider("Angular Mode lᵧ", 0, 4, st.session_state.get("l_y", 2))
    r_z = st.sidebar.slider("Radial Mode r𝓏", 0, 4, st.session_state.get("r_z", 2))
    l_z = st.sidebar.slider("Angular Mode l𝓏", 0, 4, st.session_state.get("l_z", 2))


    st.sidebar.markdown(f"**Chladni Mode Summary:**")
    st.sidebar.markdown(f"- X: r={r_x}, l={l_x}")
    st.sidebar.markdown(f"- Y: r={r_y}, l={l_y}")
    st.sidebar.markdown(f"- Z: r={r_z}, l={l_z}")

    # Use centered grid for TRR-style full symmetry
    x = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
    y = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
    z = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
if use_chladni and not use_chladni_override:
    # Use positive-only grid for traditional cymatic plate alignment
    x = np.linspace(0, domain_scale, grid_size)
    y = np.linspace(0, domain_scale, grid_size)
    z = np.linspace(0, domain_scale, grid_size)

    # Update Chladni-derived frequency and phase
    log_fx, phase_x_deg = chladni_mode_to_waveparams(r_x, l_x, 'x')
    log_fy, phase_y_deg = chladni_mode_to_waveparams(r_y, l_y, 'y')
    log_fz, phase_z_deg = chladni_mode_to_waveparams(r_z, l_z, 'z')

    fx = 10 ** log_fx
    fy = 10 ** log_fy
    fz = 10 ** log_fz

    phase_x = np.radians(phase_x_deg)
    phase_y = np.radians(phase_y_deg)
    phase_z = np.radians(phase_z_deg)

else:
    # Use centered grid for TRR-style full symmetry
    x = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
    y = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
    z = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)


   

with st.sidebar:
    st.markdown("## Resources")
    with open("Rendered_Reality_TimG.pdf", "rb") as f:
        st.download_button(
            label="Download TRR White Paper",
            data=f,
            file_name="Rendered_Reality_TimG.pdf",
            mime="application/pdf"
        )

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

EX = np.sin(fx * X + phase_x)
EY = np.sin(fy * Y + phase_y)
EZ = np.sin(fz * Z + phase_z)
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
