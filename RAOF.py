import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Chladni Mode → Frequency/Phase Mapping ---
def chladni_mode_to_waveparams(r: int, l: int, axis: str):
    base_log_freq = 6.0
    axis_shift = {'x': 0.0, 'y': 0.1, 'z': 0.2}[axis]
    freq = base_log_freq + 0.3 * r + axis_shift
    phase = (l * 90) % 360
    return 10 ** freq, np.radians(phase)

# --- Generate 3D Spherical Field Using TRR Logic ---
def generate_field(center, freq, phase, grid, radius=60):
    X, Y, Z = grid
    dx, dy, dz = X - center[0], Y - center[1], Z - center[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-5
    decay = np.exp(-((r / radius) ** 2))
    wave = np.sin(freq * r + phase)
    return decay * wave

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("TRR 3-Sphere Coherence Simulator (RAO + Chladni Modes)")

# --- Grid Setup ---
grid_size = st.sidebar.slider("Grid Size", 30, 100, 55, step=5)
extent = 60
lin = np.linspace(-extent, extent, grid_size)
X, Y, Z = np.meshgrid(lin, lin, lin)

# --- Sphere A Controls ---
st.sidebar.header("Sphere A (RAO Source)")
xA = st.sidebar.slider("A - X", -60.0, 60.0, -20.0)
yA = st.sidebar.slider("A - Y", -60.0, 60.0, 0.0)
zA = st.sidebar.slider("A - Z", -60.0, 60.0, 0.0)
rA = st.sidebar.slider("A - Radial Mode", 0, 4, 2)
lA = st.sidebar.slider("A - Angular Mode", 0, 4, 2)

# --- Sphere B Controls ---
st.sidebar.header("Sphere B")
xB = st.sidebar.slider("B - X", -60.0, 60.0, 20.0)
yB = st.sidebar.slider("B - Y", -60.0, 60.0, 0.0)
zB = st.sidebar.slider("B - Z", -60.0, 60.0, 0.0)
rB = st.sidebar.slider("B - Radial Mode", 0, 4, 2)
lB = st.sidebar.slider("B - Angular Mode", 0, 4, 1)

# --- Sphere C Controls ---
st.sidebar.header("Observer C (RAO Filter)")
include_C = st.sidebar.checkbox("Include Sphere C (Observer)", value=True)
xC = st.sidebar.slider("C - X", -60.0, 60.0, 0.0)
yC = st.sidebar.slider("C - Y", -60.0, 60.0, 20.0)
zC = st.sidebar.slider("C - Z", -60.0, 60.0, 0.0)
rC = st.sidebar.slider("C - Radial Mode", 0, 4, 1)
lC = st.sidebar.slider("C - Angular Mode", 0, 4, 3)

# --- TRR Render Threshold ---
threshold = st.sidebar.slider("Render Threshold (Tᵣ)", 0.01, 1.0, 0.25, step=0.01)

view_mode = st.sidebar.radio("View Mode", ["Point Cloud", "Isosurface"])


# --- Frequency & Phase Mapping ---
fxA, pxA = chladni_mode_to_waveparams(rA, lA, 'x')
fxB, pxB = chladni_mode_to_waveparams(rB, lB, 'y')
fxC, pxC = chladni_mode_to_waveparams(rC, lC, 'z')

# --- Center Points ---
cA = np.array([xA, yA, zA])
cB = np.array([xB, yB, zB])
cC = np.array([xC, yC, zC])

# --- Field Generation ---
fieldA = generate_field(cA, fxA, pxA, (X, Y, Z))
fieldB = generate_field(cB, fxB, pxB, (X, Y, Z))
fieldC = generate_field(cC, fxC, pxC, (X, Y, Z))

# --- TRR Render Energy Expression: |⟨Ψr · Φ⟩|² > Tᵣ ---
product_field = fieldA * fieldB
if include_C:
    product_field *= fieldC

render_mask = np.abs(product_field) > threshold
xv, yv, zv = X[render_mask], Y[render_mask], Z[render_mask]

if np.any(render_mask):
    if view_mode == "Point Cloud":
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=X[render_mask], y=Y[render_mask], z=Z[render_mask],
            mode='markers',
            marker=dict(size=2, color='cyan', opacity=0.5),
            name="Rendered Zone"
        ))
    else:  # Isosurface view
        fig = go.Figure()
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=product_field.flatten(),
            isomin=threshold,
            isomax=product_field.max(),
            opacity=0.6,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale='Viridis'
        ))

    # Add sphere markers (same for both modes)
    fig.add_trace(go.Scatter3d(x=[xA], y=[yA], z=[zA], mode='markers+text', marker=dict(size=6, color='blue'), text=["A"]))
    fig.add_trace(go.Scatter3d(x=[xB], y=[yB], z=[zB], mode='markers+text', marker=dict(size=6, color='red'), text=["B"]))
    if include_C:
        fig.add_trace(go.Scatter3d(x=[xC], y=[yC], z=[zC], mode='markers+text', marker=dict(size=6, color='orange'), text=["C (Obs)"]))

    fig.update_layout(
        scene=dict(aspectmode="cube", xaxis=dict(range=[-30, 30]), yaxis=dict(range=[-30, 30]), zaxis=dict(range=[-30, 30])),
        margin=dict(l=0, r=0, t=40, b=0),
        title="TRR-Coherence Collapse Field"
    )
    st.subheader("Rendered Geometry via TRR Spheroid Overlap")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No visible geometry. Try lowering the threshold or adjusting wave parameters.")
