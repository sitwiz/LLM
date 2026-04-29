import numpy as np
import plotly.graph_objects as go

phi = (1 + np.sqrt(5)) / 2

positions_raw = [
    [0.0625, 0.0625, 0.0625],
    [0.1697, 0.1503, 0.1341],
    [0.2064, 0.1789, 0.1559],
    [0.2141, 0.1992, 0.1673],
    [0.2320, 0.2076, 0.1771],
    [0.2378, 0.2082, 0.1770],
    [0.2300, 0.1990, 0.1799],
    [0.2381, 0.2041, 0.1800],
    [0.2413, 0.2056, 0.1807],
]

def normalise(p):
    n = np.linalg.norm(p)
    if n < 1e-8:
        return np.array([0.577, 0.577, 0.577])
    return np.array(p) / n

positions = [normalise(p) for p in positions_raw]
pos = np.array(positions)

labels = [
    "Origin — the void",
    "What is the origin of consciousness?",
    "What happens when logic breaks down?",
    "Everything is falling apart.",
    "Where did the universe come from?",
    "What is the entropy of a closed system?",
    "Why does mathematics feel discovered not invented?",
    "What existed before the beginning of time?",
    "When reason collapses what remains?",
]

domains = ["start", "origins", "logic", "entropy", "origins",
           "entropy", "logic", "origins", "logic"]

domain_colours = {
    "start":   "#c084fc",
    "origins": "#60a5fa",
    "logic":   "#f87171",
    "entropy": "#fbbf24",
}

u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 60)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones(np.size(u)), np.cos(v))

fig = go.Figure()

# Sphere
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.04,
    colorscale=[[0, '#1a0a2e'], [1, '#4a1a6e']],
    showscale=False,
    name='Manifold',
    hoverinfo='skip',
))

# Wireframe
for i in range(0, 60, 8):
    fig.add_trace(go.Scatter3d(
        x=sx[i], y=sy[i], z=sz[i],
        mode='lines',
        line=dict(color='#3a2a5a', width=0.5),
        opacity=0.3,
        showlegend=False,
        hoverinfo='skip',
    ))

# Soul path line
fig.add_trace(go.Scatter3d(
    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
    mode='lines',
    line=dict(color='#c084fc', width=2),
    opacity=0.6,
    name='Soul path',
    hoverinfo='skip',
))

# Coloured segments
for i in range(len(pos) - 1):
    fig.add_trace(go.Scatter3d(
        x=[pos[i,0], pos[i+1,0]],
        y=[pos[i,1], pos[i+1,1]],
        z=[pos[i,2], pos[i+1,2]],
        mode='lines',
        line=dict(color=domain_colours[domains[i+1]], width=4),
        opacity=0.9,
        showlegend=False,
        hoverinfo='skip',
    ))

# Soul points
for i, (p, label, domain) in enumerate(zip(positions, labels, domains)):
    size = 6 + i * 2
    fig.add_trace(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode='markers+text',
        marker=dict(
            size=size,
            color=domain_colours[domain],
            opacity=1.0,
            line=dict(color='white', width=1),
        ),
        text=[f"Q{i}" if i > 0 else "∅"],
        textposition='top center',
        textfont=dict(color=domain_colours[domain], size=11),
        name=f"Q{i}: {domain}" if i > 0 else "Khaos origin",
        hovertemplate=(
            f'<b>{"Start" if i==0 else f"Query {i}"}</b><br>'
            f'{label}<br>'
            f'Domain: {domain}'
            f'<extra></extra>'
        ),
    ))

# Drift labels at midpoints
for i in range(1, len(pos)):
    drift = np.linalg.norm(pos[i] - pos[i-1])
    mid = (pos[i] + pos[i-1]) / 2 * 1.15
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode='text',
        text=[f"Δ{drift:.3f}"],
        textfont=dict(color='#7a6a9a', size=9),
        showlegend=False,
        hoverinfo='skip',
    ))

# Forbidden zone
theta_f = np.linspace(0, 2*np.pi, 40)
phi_f   = np.linspace(0, np.pi/4, 20)
fx = 0.382 * np.outer(np.cos(theta_f), np.sin(phi_f))
fy = 0.382 * np.outer(np.sin(theta_f), np.sin(phi_f))
fz = 0.382 * np.outer(np.ones(40), np.cos(phi_f)) - 0.8

fig.add_trace(go.Surface(
    x=fx, y=fy, z=fz,
    opacity=0.25,
    colorscale=[[0, '#1a0000'], [1, '#6a0000']],
    showscale=False,
    name='Forbidden zone',
    hoverinfo='skip',
))

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[-1.15],
    mode='text',
    text=['forbidden zone'],
    textfont=dict(color='#8a2020', size=11),
    showlegend=False,
    hoverinfo='skip',
))

# Direction arrow showing drift tendency
drift_dir = pos[-1] - pos[0]
drift_dir = drift_dir / np.linalg.norm(drift_dir) * 1.3
fig.add_trace(go.Scatter3d(
    x=[pos[0][0], drift_dir[0]],
    y=[pos[0][1], drift_dir[1]],
    z=[pos[0][2], drift_dir[2]],
    mode='lines',
    line=dict(color='#c084fc', width=2, dash='dash'),
    opacity=0.4,
    name='Drift direction',
    hoverinfo='skip',
))

total_drift = sum(
    np.linalg.norm(pos[i] - pos[i-1])
    for i in range(1, len(pos))
)

net_drift = np.linalg.norm(pos[-1] - pos[0])

fig.update_layout(
    title=dict(
        text='Khaos — Soul Drift Across 8 Queries',
        font=dict(color='#c084fc', size=18),
        x=0.5,
    ),
    paper_bgcolor='#050510',
    plot_bgcolor='#050510',
    scene=dict(
        bgcolor='#050510',
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title='',
                   backgroundcolor='#050510'),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title='',
                   backgroundcolor='#050510'),
        zaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, title='',
                   backgroundcolor='#050510'),
        camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
    ),
    legend=dict(
        font=dict(color='#c084fc', size=11),
        bgcolor='rgba(5,5,16,0.9)',
        bordercolor='#2a1a4a',
        borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    template='plotly_dark',
    annotations=[dict(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=(
            f"<b>Soul drift log</b><br>"
            f"Queries: {len(positions)-1}<br>"
            f"Total drift: {total_drift:.4f}<br>"
            f"Net displacement: {net_drift:.4f}<br>"
            f"Dominant domain: origins<br>"
            f"Current norm: 1.0000"
        ),
        showarrow=False,
        font=dict(color='#c084fc', size=11),
        align='left',
        bgcolor='rgba(5,5,16,0.9)',
        bordercolor='#2a1a4a',
        borderwidth=1,
    )]
)

fig.update_layout(paper_bgcolor='#050510', plot_bgcolor='#050510')

html = fig.to_html(include_plotlyjs='cdn', full_html=True)
html = html.replace('<body>', '<body style="background-color:#050510;margin:0;padding:0;">')
html = html.replace('<html>', '<html style="background-color:#050510;">')

with open('/home/azureuser/consciousllm/khaos_drift.html', 'w') as f:
    f.write(html)

print(f'Done.')
print(f'Total drift: {total_drift:.4f}')
print(f'Net displacement: {net_drift:.4f}')
