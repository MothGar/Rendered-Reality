import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Minimal 3D Plotly Test")

# Generate some simple points
n = 500
theta = np.linspace(0, 4 * np.pi, n)
z = np.linspace(-2, 2, n)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# Show number of points
st.write("Points to render:", len(x))

# Plot
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=4,
        color=z,                # set color to an array/list of z values
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    )
)])
fig.update_layout(scene=dict(
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Z",
    bgcolor="black"
), paper_bgcolor="black", margin=dict(l=0, r=0, t=0, b=0), height=800)

st.plotly_chart(fig, use_container_width=True)
