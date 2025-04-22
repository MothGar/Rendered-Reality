
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import math
from sklearn.cluster import DBSCAN

def clamp_log(value, minval=-3.0, maxval=20.0):
    try:
        val = float(value)
        if math.isnan(val):
            val = 6.0
    except:
        val = 6.0
    return max(min(val, maxval), minval)

st.set_page_config(layout="wide")
st.title("TRR Resonance Cluster Visualizer")

# Parameters
fx = 13.5
fy = 13.5
fz = 13.5
phase_x = np.radians(90)
phase_y = np.radians(90)
phase_z = np.radians(90)
domain_scale = 1.8
grid_size = 65
threshold = 0.02
lock_strength = 0.01

# Domain setup
x = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
y = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
z = np.linspace(-domain_scale / 2, domain_scale / 2, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Wave interference
EX = np.sin(fx * X + phase_x)
EY = np.sin(fy * Y + phase_y)
EZ = np.sin(fz * Z + phase_z)
interference = np.abs(EX * EY * EZ)
field_norm = (interference - interference.min()) / (interference.max() - interference.min())

# Threshold logic
lock_mask = ((field_norm > threshold - lock_strength) & (field_norm < threshold + lock_strength))
xv, yv, zv = X[lock_mask], Y[lock_mask], Z[lock_mask]
points = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T

if len(points) > 0:
    # DBSCAN clustering
    db = DBSCAN(eps=0.05, min_samples=10).fit(points)
    labels = db.labels_

    # Color assignment
    colors = []
    for label in labels:
        if label == -1:
            colors.append('gray')
        elif label % 2 == 0:
            colors.append('orange')
        else:
            colors.append('blue')

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.6
        )
    ))
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black', scene_bgcolor='black',
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No visible geometry. Try adjusting threshold or resonance settings.")
