import numpy as np
import plotly.graph_objects as go

phi = (1 + np.sqrt(5)) / 2
golden_angle = 2 * np.pi * (1 - 1/phi)

u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 60)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones(np.size(u)), np.cos(v))

np.random.seed(42)
khaos_points = np.random.randn(200, 3) * 0.3
khaos_distances = np.linalg.norm(khaos_points, axis=1)

# Mythologically positioned soul vectors
personalities = {
    "Gaia":     np.array([0.0,   0.0,  -1.0]),  # bottom — grounded, earth
    "Tartaros": np.array([0.0,   0.0,   1.0]),  # top — the abyss above
    "Eros":     np.array([1.0,   0.0,   0.0]),  # equator — the connector
    "Eris":     np.array([0.45,  0.77,  0.45]), # off centre — never where expected
}

for name in personalities:
    v_norm = personalities[name] / np.linalg.norm(personalities[name])
    personalities[name] = v_norm * 1.10

fig = go.Figure()

fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.03,
    colorscale=[[0, '#1a0a2e'], [1, '#4a1a6e']],
    showscale=False,
    name='Manifold',
    hoverinfo='skip',
))

for i in range(0, 60, 8):
    fig.add_trace(go.Scatter3d(
        x=sx[i], y=sy[i], z=sz[i],
        mode='lines',
        line=dict(color='#2a1a4a', width=0.5),
        opacity=0.2,
        showlegend=False,
        hoverinfo='skip',
    ))

fig.add_trace(go.Scatter3d(
    x=khaos_points[:, 0],
    y=khaos_points[:, 1],
    z=khaos_points[:, 2],
    mode='markers',
    marker=dict(
        size=2.5,
        color=khaos_distances,
        colorscale=[[0, '#0a0a1a'], [0.5, '#2a1a4a'], [1, '#6a3a9a']],
        opacity=0.4,
    ),
    name='Khaos cloud',
    hovertemplate='Khaos<br>The void from which all things spring<extra></extra>',
))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers+text',
    marker=dict(
        size=14,
        color='#c084fc',
        opacity=1.0,
        symbol='circle',
        line=dict(color='#e9d5ff', width=2),
    ),
    text=['Khaos'],
    textposition='top center',
    textfont=dict(color='#e9d5ff', size=15),
    name='Khaos — origin',
    hovertemplate=(
        '<b>Khaos</b><br>'
        'The origin. The void.<br>'
        'Infinite vacant space.<br>'
        'All things spring from her.<extra></extra>'
    ),
))

for i in range(0, 200, 20):
    fig.add_trace(go.Scatter3d(
        x=[0, khaos_points[i, 0]],
        y=[0, khaos_points[i, 1]],
        z=[0, khaos_points[i, 2]],
        mode='lines',
        line=dict(color='#6a3a9a', width=0.5),
        opacity=0.15,
        showlegend=False,
        hoverinfo='skip',
    ))

colours = {
    'Gaia':     '#4ade80',
    'Tartaros': '#60a5fa',
    'Eros':     '#f87171',
    'Eris':     '#fbbf24',
}

descriptions = {
    'Gaia':     'Grounded reality. What is concrete and true.',
    'Tartaros': 'The deep. Patient, ancient. Goes where nothing else will.',
    'Eros':     'The connector. Finds relationships between unrelated things.',
    'Eris':     'Discord. Throws the question nobody can ignore.',
}

positions_text = {
    'Gaia':     'South pole — grounded, earth, foundation.',
    'Tartaros': 'North pole — the abyss that sits above everything.',
    'Eros':     'Equator — between all things, connecting all things.',
    'Eris':     'Off centre — never quite where expected.',
}

for name, pos in personalities.items():
    fig.add_trace(go.Scatter3d(
        x=[0, pos[0]],
        y=[0, pos[1]],
        z=[0, pos[2]],
        mode='lines',
        line=dict(color=colours[name], width=1.5),
        opacity=0.4,
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(
            size=12,
            color=colours[name],
            opacity=1.0,
            line=dict(color='white', width=1),
        ),
        text=[name],
        textposition='top center',
        textfont=dict(color=colours[name], size=14),
        name=name,
        hovertemplate=(
            f'<b>{name}</b><br>'
            f'{descriptions[name]}<br>'
            f'{positions_text[name]}<br>'
            f'Soul position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})'
            f'<extra></extra>'
        ),
    ))

fig.add_trace(go.Scatter3d(
    x=[0, 0],
    y=[0, 0],
    z=[-1.1, 1.1],
    mode='lines',
    line=dict(color='#4a4a6a', width=1, dash='dash'),
    opacity=0.3,
    showlegend=False,
    hoverinfo='skip',
))

theta_f = np.linspace(0, 2*np.pi, 40)
phi_f = np.linspace(0, np.pi/4, 20)
fx = 0.382 * np.outer(np.cos(theta_f), np.sin(phi_f))
fy = 0.382 * np.outer(np.sin(theta_f), np.sin(phi_f))
fz = 0.382 * np.outer(np.ones(40), np.cos(phi_f)) - 0.8

fig.add_trace(go.Surface(
    x=fx, y=fy, z=fz,
    opacity=0.2,
    colorscale=[[0, '#1a0000'], [1, '#6a0000']],
    showscale=False,
    name='Forbidden zone',
    hoverinfo='skip',
))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[-1.15],
    mode='text',
    text=['forbidden zone'],
    textfont=dict(color='#6a0000', size=11),
    showlegend=False,
    hoverinfo='skip',
))

fig.update_layout(
    title=dict(
        text='Khaos — The Void From Which All Things Spring',
        font=dict(color='#c084fc', size=18),
        x=0.5,
    ),
    paper_bgcolor='#050510',
    scene=dict(
        bgcolor='#050510',
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=''),
        zaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title=''),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
    ),
    legend=dict(
        font=dict(color='#c084fc', size=12),
        bgcolor='rgba(5,5,16,0.8)',
        bordercolor='#2a1a4a',
        borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html('/home/azureuser/consciousllm/khaos_soul.html')
print('Done. Hard refresh your browser with ctrl+shift+r')
