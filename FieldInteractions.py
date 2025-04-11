import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("TRR Field–Field Interaction Simulator")

st.markdown("""
This simulator models the **interaction between two resonance fields**, Ψᵣ and Φ, and computes the render energy using the TRR framework.

Rendering condition:
> **|⟨Ψᵣ | Φ⟩|² > Tᵣ**

You can explore how frequency and phase affect resonance strength and rendering.
""")

# --- Sidebar Controls ---
st.sidebar.title("Field Parameters")

grid_size = st.sidebar.slider("Grid Size", 20, 100, 50, step=5)
domain = st.sidebar.slider("Domain Size", 1.0, 10.0, 2.0, step=0.1)

# Ψᵣ parameters
st.sidebar.subheader("Ψᵣ – Internal Field")
fx_r = st.sidebar.slider("X Freq Ψᵣ (log₁₀ Hz)", 3.0, 9.0, 6.0, 0.1)
fy_r = st.sidebar.slider("Y Freq Ψᵣ (log₁₀ Hz)", 3.0, 9.0, 6.0, 0.1)
fz_r = st.sidebar.slider("Z Freq Ψᵣ (log₁₀ Hz)", 3.0, 9.0, 6.0, 0.1)
px_r = st.sidebar.slider("Phase Ψᵣ (°)", 0, 360, 90, 10)

# Φ parameters
st.sidebar.subheader("Φ – External Field")
fx_e = st.sidebar.slider("X Freq Φ (log₁₀ Hz)", 3.0, 9.0, 6.2, 0.1)
fy_e = st.sidebar.slider("Y Freq Φ (log₁₀ Hz)", 3.0, 9.0, 6.2, 0.1)
fz_e = st.sidebar.slider("Z Freq Φ (log₁₀ Hz)", 3.0, 9.0, 6.2, 0.1)
px_e = st.sidebar.slider("Phase Φ (°)", 0, 360, 90, 10)

# Render threshold
T_r = st.sidebar.number_input("Render Threshold Tᵣ", min_value=0.0, value=1e5, step=1e3, format="%.1f")

# Create grid
x = np.linspace(-domain / 2, domain / 2, grid_size)
y = np.linspace(-domain / 2, domain / 2, grid_size)
z = np.linspace(-domain / 2, domain / 2, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Convert log frequencies to Hz
fxr, fyr, fzr = 10**fx_r, 10**fy_r, 10**fz_r
fxe, fye, fze = 10**fx_e, 10**fy_e, 10**fz_e

# Convert phases
pxr = np.radians(px_r)
pxe = np.radians(px_e)

# Define fields
Psi_r = np.sin(fxr * X + pxr) * np.sin(fyr * Y + pxr) * np.sin(fzr * Z + pxr)
Phi   = np.sin(fxe * X + pxe) * np.sin(fye * Y + pxe) * np.sin(fze * Z + pxe)

# Interaction and render energy
interaction = Psi_r * Phi
render_energy = np.abs(np.sum(interaction))**2
is_rendered = render_energy > T_r

st.markdown(f"""
### Results

- **Render Energy:** `{render_energy:.2f}`
- **Threshold Tᵣ:** `{T_r:.2f}`
- **Rendering Occurs?** `{'✅ YES' if is_rendered else '❌ NO'}`
""")

# Normalize the overlap field
field_norm = (interaction - np.min(interaction)) / (np.max(interaction) - np.min(interaction) + 1e-9)
main_mask = field_norm > 0.7  # Standard visible geometry

# Compute all points for the main cloud
xv, yv, zv = X[main_mask], Y[main_mask], Z[main_mask]
cv = field_norm[main_mask]

# Size scaling
max_size = 18
min_size = 2
sizes = min_size + (cv - cv.min()) / (cv.max() - cv.min() + 1e-9) * (max_size - min_size)

fig = go.Figure()

# Plot main field
fig.add_trace(go.Scatter3d(
    x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
    mode='markers',
    marker=dict(
        size=sizes.flatten(),
        color=cv,
        colorscale="Viridis",
        opacity=0.65,
    ),
    name="Resonance Field"
))

# Show strong rendering zone if render condition is met
if is_rendered:
    # Define highlight: top ~5% of values
    highlight_cutoff = 0.95 * np.max(field_norm)
    high_mask = field_norm >= highlight_cutoff

    xh, yh, zh = X[high_mask], Y[high_mask], Z[high_mask]

    fig.add_trace(go.Scatter3d(
        x=xh.flatten(), y=yh.flatten(), z=zh.flatten(),
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            opacity=0.9,
            symbol='circle'
        ),
        name="Render Zones"
    ))

# Layout
fig.update_layout(
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    title="Overlap Zone Visualization — Resonant Realization Zones in Red",
    margin=dict(l=0, r=0, t=40, b=0),
    height=700
)

st.plotly_chart(fig, use_container_width=True)
