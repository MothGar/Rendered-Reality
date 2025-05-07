import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Physical constants for plasma frequency
e = 1.602e-19       # Elementary charge (C)
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
m_e = 9.109e-31     # Electron mass (kg)

# --- Generalized TRR Field Generator ---
def generate_field(center, freq, phase, grid, radius=60, mode="radial", helicity=6.0, kvec=None):
    X, Y, Z = grid
    cx, cy, cz = center
    phase_rad = np.radians(phase)

    if mode == "radial":
        r = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) + 1e-5
        wave = np.sin(freq * r + phase_rad)

    elif mode == "linear":
        if kvec is None:
            kvec = np.array([1.0, 0.0, 0.0])
        kx, ky, kz = kvec
        kdotr = kx * X + ky * Y + kz * Z
        wave = np.sin(freq * kdotr + phase_rad)

    elif mode == "helical":
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        theta = np.arctan2(dy, dx)
        helix_phase = helicity * theta + freq * dz
        wave = np.sin(helix_phase + phase_rad)

    else:
        wave = np.zeros_like(X)

    decay = np.exp(-(((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / radius**2))
    return decay * wave

# --- Setup ---
st.set_page_config(layout="wide")
st.title("TRR Plasma-Threshold Resonance Simulator")

# --- Grid Setup ---
grid_size = 100
extent = 60
lin = np.linspace(-extent, extent, grid_size)
X, Y, Z = np.meshgrid(lin, lin, lin, indexing='xy')



# --- Sidebar Controls ---
st.sidebar.header("Sphere A")
xA = st.sidebar.slider("A - X", -60.0, 60.0, -15.0)
yA = st.sidebar.slider("A - Y", -60.0, 60.0, 0.0)
zA = st.sidebar.slider("A - Z", -60.0, 60.0, 0.0)
freqA = st.sidebar.slider("A - Frequency", 0.1, 5.0, 1.80)
phaseA = st.sidebar.slider("A - Phase", 0, 360, 0)
mode_A = st.sidebar.selectbox("A - Mode", ["radial", "linear", "helical"])
kvec_A = np.array([st.sidebar.slider("A - kx", -1.0, 1.0, 1.0),
                   st.sidebar.slider("A - ky", -1.0, 1.0, 0.0),
                   st.sidebar.slider("A - kz", -1.0, 1.0, 0.0)])
helicity_A = st.sidebar.slider("A - Helicity", 0.0, 12.0, 0.0)

st.sidebar.header("Sphere B")
xB = st.sidebar.slider("B - X", -60.0, 60.0, 15.0)
yB = st.sidebar.slider("B - Y", -60.0, 60.0, 0.0)
zB = st.sidebar.slider("B - Z", -60.0, 60.0, 0.0)
freqB = st.sidebar.slider("B - Frequency", 0.1, 5.0, 1.80)
phaseB = st.sidebar.slider("B - Phase", 0, 360, 120)
mode_B = st.sidebar.selectbox("B - Mode", ["radial", "linear", "helical"])
kvec_B = np.array([st.sidebar.slider("B - kx", -1.0, 1.0, 0.0),
                   st.sidebar.slider("B - ky", -1.0, 1.0, 1.0),
                   st.sidebar.slider("B - kz", -1.0, 1.0, 0.0)])
helicity_B = st.sidebar.slider("B - Helicity", 0.0, 12.0, 8.0)
include_B = st.sidebar.checkbox("Include Sphere B", value=True)

st.sidebar.header("Sphere C (Observer)")
xC = st.sidebar.slider("C - X", -60.0, 60.0, 0.0)
yC = st.sidebar.slider("C - Y", -60.0, 60.0, 20.0)
zC = st.sidebar.slider("C - Z", -60.0, 60.0, 0.0)
freqC = st.sidebar.slider("C - Frequency", 0.1, 5.0, 1.80)
phaseC = st.sidebar.slider("C - Phase", 0, 360, 240)
mode_C = st.sidebar.selectbox("C - Mode", ["radial", "linear", "helical"])
kvec_C = np.array([st.sidebar.slider("C - kx", -1.0, 1.0, 0.0),
                   st.sidebar.slider("C - ky", -1.0, 1.0, 0.0),
                   st.sidebar.slider("C - kz", -1.0, 1.0, 1.0)])
helicity_C = st.sidebar.slider("C - Helicity", 0.0, 12.0, 0.0)
include_C = st.sidebar.checkbox("Include Sphere C", value=True)

view_mode = st.sidebar.radio("Viewer Mode", ["3D Points", "Isosurface"])
threshold_scale = st.sidebar.slider("Plasma Threshold Scale", 0.0, 1.0, 0.3)

# --- Compute Fields ---
centerA = np.array([xA, yA, zA])
centerB = np.array([xB, yB, zB])
centerC = np.array([xC, yC, zC])

fieldA = generate_field(centerA, freqA, phaseA, (X, Y, Z), 60, mode_A, helicity_A, kvec_A)
fieldB = generate_field(centerB, freqB, phaseB, (X, Y, Z), 60, mode_B, helicity_B, kvec_B)
fieldC = generate_field(centerC, freqC, phaseC, (X, Y, Z), 60, mode_C, helicity_C, kvec_C)

if include_B and include_C:
    overlap = fieldA * fieldB * fieldC
elif include_B:
    overlap = fieldA * fieldB
elif include_C:
    overlap = fieldA * fieldC
else:
    overlap = fieldA

# --- Plasma Threshold Field ---
ne_field = 1e18 * np.exp(-((X**2 + Y**2 + Z**2) / (40**2)))  # in electrons/m³
fp_field = (1 / (2 * np.pi)) * np.sqrt((ne_field * e**2) / (epsilon_0 * m_e))  # Hz
fp_scaled = (fp_field / fp_field.max()) * threshold_scale

# --- Rendering Condition ---
render_zone = (np.abs(overlap)**2 > fp_scaled)

# --- Visualization ---
fig = go.Figure()

def add_transparent_sphere(fig, center, radius=30, opacity=0.2, color="blue"):
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name="Sphere"
    ))

if view_mode == "3D Points":
    xv, yv, zv = X[render_zone], Y[render_zone], Z[render_zone]
    if xv.size > 0:
        fig.add_trace(go.Scatter3d(x=xv.flatten(), y=yv.flatten(), z=zv.flatten(),
                                   mode='markers', marker=dict(size=2, color='cyan'), name="Rendered"))
    else:
        st.warning("No points passed the threshold — try lowering the plasma scale.")
else:
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=overlap.flatten(),
        isomin=fp_scaled.min(),
        isomax=fp_scaled.max(),
        surface_count=1,
        opacity=0.8,
        colorscale="Viridis",
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="Rendered Isosurface"
    ))

# Adding Transparent Spheres
add_transparent_sphere(fig, [xA, yA, zA], radius=30, opacity=0.2, color="cyan")
if include_B:
    add_transparent_sphere(fig, [xB, yB, zB], radius=30, opacity=0.2, color="red")
if include_C:
    add_transparent_sphere(fig, [xC, yC, zC], radius=30, opacity=0.2, color="green")

fig.update_layout(scene=dict(
    aspectmode="cube",  # Enforces equal scaling for all axes
    xaxis=dict(range=[-extent, extent]),
    yaxis=dict(range=[-extent, extent]),
    zaxis=dict(range=[-extent, extent])
), margin=dict(l=0, r=0, t=60, b=0),
    title="Plasma-Constrained Resonance Geometry")


st.plotly_chart(fig, use_container_width=True)


