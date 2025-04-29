# ------------------------------------------------------------
# TRR Resonant-Sphere Simulator (fast cached version)
# ------------------------------------------------------------
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import spherical_jn, sph_harm
from scipy import special                       # Bessel zeros

# ---------- utilities ----------------------------------------------------
MAX_L = 6
zeros_jl = {l: special.jn_zeros(l, 5) for l in range(MAX_L + 1)}

def spherical_mode(n, l, m, R, grid):
    X, Y, Z = grid
    r  = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
    th = np.arccos(np.clip(Z / r, -1.0, 1.0))
    ph = np.arctan2(Y, X)
    k_nl = zeros_jl[l][n-1] / R
    field = spherical_jn(l, k_nl * r) * sph_harm(m, l, ph, th)
    field = field.real
    field[r > R] = 0.0
    return field

@st.cache_resource
def cached_mode(n, l, m, R, N):
    lin = np.linspace(-R, R, N)
    grid = np.meshgrid(lin, lin, lin, indexing="ij")
    return spherical_mode(n, l, m, R, grid)

# ---------- UI -----------------------------------------------------------
st.set_page_config(layout="wide")
st.title("TRR Resonant-Sphere Simulator")

with st.sidebar:
    st.header("Mode selection")
    n = st.slider("Radial index n", 1, 3, 1)
    l = st.slider("Angular index l", 0, 4, 2)
    m = 0 if l == 0 else st.slider("m (-l … l)", -l, l, 0)
    phase_deg = st.slider("Common phase (°)", 0, 360, 0)
    phase_rad = np.radians(phase_deg)
    R = st.slider("Sphere radius R", 20.0, 60.0, 36.0)
    dk_tol = st.slider("Δk tolerance (RAO)", 0.0, 1.0, 0.40)
    alpha = 1 / st.slider("Lock (steepness)", 0.02, 0.20, 0.10)
    eta   = st.slider("Gain η",    0.0, 5.0, 1.30)
    kappa = st.slider("Damping κ", 0.0, 0.10, 0.02)
    view  = st.radio("Viewer", ["3-D points", "Isosurface"])
    iso_pt= st.slider("Point isovalue", 0.50, 0.99, 0.95)

# ---------- grid & mode --------------------------------------------------
Ngrid = 100
lin = np.linspace(-60, 60, Ngrid)
X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

field = cached_mode(n, l, m, R, Ngrid)
field *= np.cos(phase_rad)                    # simple phase shift

# ---------- RAO filter ---------------------------------------------------
if dk_tol < 0.99:
    Fx = np.fft.fftn(field)
    kx = np.fft.fftfreq(Ngrid, d=lin[1]-lin[0]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag   = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_tgt  = zeros_jl[l][n-1] / R
    mask_k = np.abs(kmag - k_tgt) < dk_tol
    field  = np.fft.ifftn(Fx * mask_k).real

# ---------- simple gain-damp --------------------------------------------
if "prev" not in st.session_state:
    st.session_state.prev = np.zeros_like(field)
field = (2-kappa)*field - (1-kappa)*st.session_state.prev + eta*field
st.session_state.prev = field.copy()

# ---------- probability mask --------------------------------------------
T_r = 0.20
P = 1 / (1 + np.exp(-alpha * (field**2 - T_r)))
rng = np.random.default_rng(42)
mask = (P > iso_pt) & (rng.random(field.shape) < 0.02)
r = np.sqrt(X**2 + Y**2 + Z**2)               # for colour

# ---------- visualisation -----------------------------------------------
fig = go.Figure()

if view == "3-D points":
    if mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=X[mask], y=Y[mask], z=Z[mask],
                mode="markers",
                marker=dict(size=3, opacity=0.7,
                            color=r[mask], colorscale="Turbo"),
                name="voxels"))
    else:
        st.warning("No voxels passed the cut.")

else:  # isosurface
    abs_max = np.abs(field).max()
    # + lobe
    fig.add_trace(go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=field.ravel(),
        isomin=+0.05*abs_max, isomax=abs_max,
        opacity=0.5, colorscale="Viridis", name="+"))
    # - lobe
    neg_peak = np.abs(field[field < 0]).max()
    fig.add_trace(
        go.Isosurface(
            x=X.ravel(),  y=Y.ravel(),  z=Z.ravel(),
            value=field.ravel(),
            isomin=-neg_peak,
            isomax=-.02*neg_peak,   # 2 % instead of 15 %
            surface_count=1,
            opacity=0.8, colorscale="Plasma", name="- lobe",
        caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

fig.update_layout(scene=dict(aspectmode="cube"),
                  margin=dict(l=20, r=20, t=40, b=0),
                  height=700,
                  title=f"n={n}, l={l}, m={m} | η={eta:.2f}, κ={kappa:.2f}")
st.plotly_chart(fig, use_container_width=True)
