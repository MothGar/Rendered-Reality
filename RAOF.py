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
st.title("TRR Resonant-Sphere Simulator with Tier-Coupling")
if "apply_triggered" not in st.session_state:
    st.session_state.apply_triggered = False

apply_clicked = st.sidebar.button("✅ Apply Changes")
if apply_clicked:
    st.session_state.apply_triggered = True

# ========== Independent Sphere Parameters ==========
with st.sidebar.expander("🔴 Sphere A — Central", expanded=True):
    n_A = st.slider("n (A)", 1, 3, 1, key="nA")
    l_A = st.slider("l (A)", 0, 4, 2, key="lA")
    m_A = 0 if l_A == 0 else st.slider("m (A)", -l_A, l_A, 0, key="mA")
    R_A = st.slider("Radius R (A)", 20.0, 60.0, 36.0, key="RA")

with st.sidebar.expander("🔵 Sphere B — X Offset", expanded=True):
    n_B = st.slider("n (B)", 1, 3, 1, key="nB")
    l_B = st.slider("l (B)", 0, 4, 2, key="lB")
    m_B = 0 if l_B == 0 else st.slider("m (B)", -l_B, l_B, 0, key="mB")
    R_B = st.slider("Radius R (B)", 20.0, 60.0, 36.0, key="RB")

with st.sidebar.expander("🟢 Sphere C — Y Offset", expanded=True):
    n_C = st.slider("n (C)", 1, 3, 1, key="nC")
    l_C = st.slider("l (C)", 0, 4, 2, key="lC")
    m_C = 0 if l_C == 0 else st.slider("m (C)", -l_C, l_C, 0, key="mC")
    R_C = st.slider("Radius R (C)", 20.0, 60.0, 36.0, key="RC")

# ========== Grid Setup ==========
Ngrid = 100
lin = np.linspace(-60, 60, Ngrid)
X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

# Sphere positions
offset_A = np.array([0.0, 0.0, 0.0])
offset_B = np.array([60.0, 0.0, 0.0])
offset_C = np.array([0.0, 60.0, 0.0])

# ========== Mode Calculations ==========
field_A = spherical_mode(n_A, l_A, m_A, R_A, (X - offset_A[0], Y - offset_A[1], Z - offset_A[2]))
field_B = spherical_mode(n_B, l_B, m_B, R_B, (X - offset_B[0], Y - offset_B[1], Z - offset_B[2]))
field_C = spherical_mode(n_C, l_C, m_C, R_C, (X - offset_C[0], Y - offset_C[1], Z - offset_C[2]))

# Combine fields (average or weighted sum if desired)
dk_tol = st.slider("Δk tolerance (RAO)", 0.0, 1.0, 0.30)
alpha = 1 / st.slider("Lock (steepness)", 0.02, 0.20, 0.10)
eta   = st.slider("Gain η",    0.0, 5.0, 1.30)
kappa = st.slider("Damping κ", 0.0, 0.10, 0.02)
iso_pt = st.slider("Point isovalue", 0.50, 0.99, 0.95)
field = (field_A + field_B + field_C) / 3.0

# ---------- RAO filter ---------------------------------------------------
if dk_tol < 0.20:
    Fx = np.fft.fftn(field)
    kx = np.fft.fftfreq(Ngrid, d=lin[1]-lin[0]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag   = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_tgt = zeros_jl[l_A][n_A - 1] / R_A
    mask_k = np.abs(kmag - k_tgt) < dk_tol
    field  = np.fft.ifftn(Fx * mask_k).real

# ---------- simple gain-damp --------------------------------------------
if "prev" not in st.session_state:
    st.session_state.prev = np.zeros_like(field)
field = (2-kappa)*field - (1-kappa)*st.session_state.prev + eta*field
st.session_state.prev = field.copy()

# ---------- dynamic threshold -------------------------------------------
# baseline threshold
T_r0 = st.slider("Baseline threshold T_r0", 0.0, 1.0, 0.20, step=0.01)
# compute time-dependent render threshold
# Governing layer coupling
alpha_CG = st.slider("G-layer coupling α_CG", 0.0, 2.0, 0.5, step=0.05)
B_G = st.slider("G-layer bias B_G", -1.0, 1.0, 0.0, step=0.01)
# Cognitive layer coupling
alpha_CC = st.slider("C-layer coupling α_CC", 0.0, 2.0, 0.5, step=0.05)
B_C = st.slider("C-layer bias B_C", -1.0, 1.0, 0.0, step=0.01)

T_r = T_r0 - alpha_CG * B_G - alpha_CC * B_C

# ---------- probability mask --------------------------------------------
P = 1 / (1 + np.exp(-alpha * (field**2 - T_r)))
rng = np.random.default_rng(42)
mask = (P > iso_pt) & (rng.random(field.shape) < 0.02)
r = np.sqrt(X**2 + Y**2 + Z**2)               # for colour
view = st.radio("Viewer", ["3-D points", "Isosurface"])

# Apply-triggered rendering block
if st.session_state.get("apply_triggered", False):

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

    elif view == "Isosurface":
        abs_max = np.abs(field).max()
        fig.add_trace(
            go.Isosurface(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                value=field.ravel(),
                isomin=+0.02 * abs_max, isomax=abs_max,
                opacity=0.80,
                colorscale=[[0.0, "rgb(255,180,0)"],
                            [1.0, "rgb(255,100,0)"]],
                name="+ lobe",
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )

        neg_slice = field[field < 0]
        if neg_slice.size:
            neg_peak = np.abs(neg_slice).max()
            fig.add_trace(
                go.Isosurface(
                    x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                    value=field.ravel(),
                    isomin=-neg_peak, isomax=-0.02 * neg_peak,
                    opacity=0.30,
                    colorscale="Plasma",
                    name="- lobe",
                    caps=dict(x_show=False, y_show=False, z_show=False),
                )
            )
        else:
            st.info("No negative field values in this frame – skipped ‘− lobe’.")

    fig.update_layout(scene=dict(aspectmode="cube"),
                      margin=dict(l=20, r=20, t=40, b=0),
                      height=700,
                      title=f"A(n={n_A}, l={l_A}, m={m_A}) | B(n={n_B}, l={l_B}, m={m_B}) | C(n={n_C}, l={l_C}, m={m_C}) | T_r={T_r:.2f}")
    st.plotly_chart(fig, use_container_width=True)

    # Reset trigger
    st.session_state.apply_triggered = False
