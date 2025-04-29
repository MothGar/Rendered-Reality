# ------------------------------------------------------------
# TRR Resonant-Sphere Simulator  –  CWWE + RAO reference build
# ------------------------------------------------------------
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import spherical_jn, sph_harm
from scipy import special                       # Bessel zeros

# ---------- 1. utilities -------------------------------------------------
MAX_L = 6                                       # raise if you want higher l
zeros_jl = {l: special.jn_zeros(l, 5) for l in range(MAX_L + 1)}  # pre-tabulate

def spherical_mode(n: int, l: int, m: int, R: float, grid):
    """
    Real part of the (n,l,m) spherical eigen-mode inside a hard sphere radius R.
    n starts at 1 (first radial zero).
    """
    X, Y, Z = grid
    r  = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-12
    th = np.arccos(np.clip(Z / r, -1.0, 1.0))
    ph = np.arctan2(Y, X)

    k_nl = zeros_jl[l][n - 1] / R               # wave-number
    field = spherical_jn(l, k_nl * r) * sph_harm(m, l, ph, th)
    field = field.real
    field[r > R] = 0.0                          # zero outside sphere
    return field

# ---------- 2. Streamlit page / sidebar ----------------------------------
st.set_page_config(layout="wide")
st.title("TRR Resonant-Sphere Simulator")

st.sidebar.header("Mode selection")
n = st.sidebar.slider("Radial index  n", 1, 3, 1)
l = st.sidebar.slider("Angular index l", 0, 4, 2)

if l == 0:
    m = 0
    st.sidebar.write("m = 0 (only value for l = 0)")
else:
    m = st.sidebar.slider("m  (-l … l)", -l, l, 0)

phase_deg = st.sidebar.slider("Common phase (°)", 0, 360, 0)
phase_rad = np.radians(phase_deg)
domain_R  = st.sidebar.slider("Sphere radius R (grid units)", 20.0, 60.0, 36.0)

dk_allowed = st.sidebar.slider("Δk tolerance  (RAO filter)", 0.0, 1.0, 0.4)
alpha_lock = 1.0 / st.sidebar.slider("Lock  (steepness)", 0.02, 0.20, 0.10)

eta   = st.sidebar.slider("Gain  η",    0.0, 5.0,   1.3)
kappa = st.sidebar.slider("Damping κ",  0.0, 0.10,  0.04)

view_mode = st.sidebar.radio("Viewer mode", ["3D Points", "Isosurface"])

# ---------- 3. Grid ------------------------------------------------------
grid_size = 100
extent = 60.0
lin = np.linspace(-extent, extent, grid_size)
X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")

# ---------- 4. Build carrier field --------------------------------------
field = spherical_mode(n, l, m, domain_R, (X, Y, Z))
field = np.cos(phase_rad) * field - np.sin(phase_rad) * field   # global phase shift

# ----- optional k-space RAO filter (keeps modes within dk_allowed) -------
Fx   = np.fft.fftn(field)
kx   = np.fft.fftfreq(grid_size, d=lin[1] - lin[0]) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
k_mag    = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
k_target = zeros_jl[l][n - 1] / domain_R
mask = np.abs(k_mag - k_target) < dk_allowed
field = np.fft.ifftn(Fx * mask).real

# ---------- 5. Simple gain–damping update (one time step) ---------------
if "field_prev" not in st.session_state:
    st.session_state.field_prev = np.zeros_like(field)

field_next = (2 - kappa) * field - (1 - kappa) * st.session_state.field_prev + eta * field
st.session_state.field_prev = field.copy()
field = field_next

# ---------- 6. Logistic render probability + voxel mask ---------------
T_r = 0.20                                          # render threshold
iso = st.sidebar.slider("Point isovalue", 0.50, 0.99, 0.95)

Prender = 1.0 / (1.0 + np.exp(-alpha_lock * (field**2 - T_r)))

rng   = np.random.default_rng(42)
mask  = (Prender > iso)                            # keep only high-prob voxels
mask &= rng.random(field.shape) < 0.02             # 2 % subsample

r = np.sqrt(X**2 + Y**2 + Z**2)                    # radial coord for colouring



# ---------- 7. Visualisation ------------------------------------------
fig = go.Figure()

if view_mode == "3D Points":
    if mask.any():
        fig.add_trace(
            go.Scatter3d(
                x = X[mask],  y = Y[mask],  z = Z[mask],
                mode   = "markers",
                marker = dict(
                    size       = 3,
                    opacity    = 0.7,
                    color      = r[mask],           # colour-gradient
                    colorscale = "Turbo",
                ),
                name = "Rendered voxels",
            )
        )
    else:
        st.warning("No voxels rendered – raise η or lower Lock / threshold.")
else:
abs_max = np.abs(field).max()

fig.add_trace(
    go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=field.flatten(),
        isomin=+0.3*abs_max, isomax=abs_max,
        surface_count=1, opacity=0.6,
        colorscale="Viridis", name="+ lobe",
        caps=dict(x_show=False, y_show=False, z_show=False)
    )
)
fig.add_trace(
    go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=field.flatten(),
        isomin=-abs_max, isomax=-0.3*abs_max,
        surface_count=1, opacity=0.6,
        colorscale="Plasma", name="- lobe",
        caps=dict(x_show=False, y_show=False, z_show=False)
    )
)


fig.update_layout(
    scene=dict(aspectmode="cube"),
    margin=dict(l=20, r=20, t=40, b=0),
    title=f"n={n}, l={l}, m={m} | η={eta:.2f}, κ={kappa:.2f}, Lock={1/alpha_lock:.3f}",
    height = 700
)
st.plotly_chart(fig, use_container_width=True)
