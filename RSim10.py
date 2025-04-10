import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# --- TRR White Paper Download ---
with st.sidebar:
    st.markdown("## Resources")
    try:
        with open("Rendered_Reality_TimG.pdf", "rb") as f:
            st.download_button(
                label="Download TRR White Paper",
                data=f,
                file_name="Rendered_Reality_TimG.pdf",
                mime="application/pdf"
            )
    except:
        st.warning("TRR white paper not found.")

# --- Equation Explanation ---
st.markdown("""
---
**TRR Equation of Rendered Geometry**

> **Render Condition:**  
> \[ |⟨ \Psi_r(x, t) | H_{res} | \Phi(x, t) ⟩|^2 > T_r \]

This simulator visualizes the points in 3D space where this equation **crosses the render threshold**—where reality takes form.
""")

# --- TRR Conceptual Explanation ---
with st.expander("📘 What Is TRR Isoplane Geometry?"):
    st.markdown("""
    **TRR (Theory of Rendered Reality)** models how reality appears when wave-based resonance fields align into coherent patterns.

    **Isoplane Geometry** happens when X, Y, and Z waves intersect *just right*—forming stable structures.

    **Simplified Equation:**  
    `resonance_energy > threshold`  
    _When internal and external fields overlap enough, reality 'renders'._

    This viewer shows you **where** that happens in space.

    - High alignment = more structure  
    - Phase mismatch = less pattern  
    - Thresholds let you filter only the coherent zones
    """)

# --- Chladni Mode Functions ---
def chladni_mode_to_waveparams(r: int, l: int, axis: str):
    base_log_freq = 6.0
    axis_shift = {'x': 0.0, 'y': 0.1, 'z': 0.2}[axis]
    freq = base_log_freq + 0.3 * r + axis_shift
    phase = (l * 90) % 360
    return freq, phase

# --- Chladni Mode Input Section ---
use_chladni = st.sidebar.checkbox("Enable Chladni Mode Input")

if use_chladni:
    st.sidebar.markdown("### Chladni Mode Selection")

    r_x = st.sidebar.slider("Radial Mode rₓ", 0, 4, 2)
    l_x = st.sidebar.slider("Angular Mode lₓ", 0, 4, 1)
    r_y = st.sidebar.slider("Radial Mode rᵧ", 0, 4, 2)
    l_y = st.sidebar.slider("Angular Mode lᵧ", 0, 4, 2)
    r_z = st.sidebar.slider("Radial Mode r𝓏", 0, 4, 2)
    l_z = st.sidebar.slider("Angular Mode l𝓏", 0, 4, 3)

    log_fx, phase_x_deg = chladni_mode_to_waveparams(r_x, l_x, 'x')
    log_fy, phase_y_deg = chladni_mode_to_waveparams(r_y, l_y, 'y')
    log_fz, phase_z_deg = chladni_mode_to_waveparams(r_z, l_z, 'z')

    fx = 10 ** log_fx
    fy = 10 ** log_fy
    fz = 10 ** log_fz

    phase_x = np.radians(phase_x_deg)
    phase_y = np.radians(phase_y_deg)
    phase_z = np.radians(phase_z_deg)

    st.sidebar.markdown(f"**Chladni Mode Summary:**")
    st.sidebar.markdown(f"- X: r={r_x}, l={l_x} → f={log_fx:.2f}, ϕ={phase_x_deg}°")
    st.sidebar.markdown(f"- Y: r={r_y}, l={l_y} → f={log_fy:.2f}, ϕ={phase_y_deg}°")
    st.sidebar.markdown(f"- Z: r={r_z}, l={l_z} → f={log_fz:.2f}, ϕ={phase_z_deg}°")

    domain_scale = 10.0
    grid_size = 40
    threshold = 0.05
    lock_strength = 0.01

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

    if len(xv) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
            mode='markers',
            marker=dict(size=2, color=color_vals, colorscale='Viridis', opacity=0.6)
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
