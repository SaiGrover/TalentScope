"""
EDA Module — returns Plotly JSON specs (rendered client-side via Plotly CDN)
No server-side plotly dependency needed.
"""
import pandas as pd
import numpy as np
import json


# ── Shared Plotly theme ────────────────────────────────────────────────────────
BG      = '#0A0E1A'
PANEL   = '#141B2D'
PANEL2  = '#1C2640'
RIM     = '#1E2D4A'
TEXT    = '#E8EDF5'
MUTED   = '#6B7FA3'
ACCENT  = '#4F8EF7'
VIOLET  = '#8B5CF6'
EMERALD = '#10B981'
AMBER   = '#F59E0B'
ROSE    = '#F43F5E'
CYAN    = '#06B6D4'

PALETTE = [ACCENT, VIOLET, EMERALD, AMBER, ROSE, CYAN,
           '#F97316', '#84CC16', '#EC4899', '#14B8A6']

def _layout(title='', height=380, **extra):
    base = dict(
        title=dict(text=title, font=dict(family='Syne,sans-serif', size=15, color=TEXT), x=0.02),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL2,
        font=dict(family='DM Sans,sans-serif', color=MUTED, size=11),
        height=height,
        margin=dict(l=48, r=20, t=48, b=48),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=10)),
        xaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
    )
    base.update(extra)
    return base


def _j(fig_dict):
    """Serialize to JSON-safe dict."""
    return json.loads(json.dumps(fig_dict, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))


def perform_eda(df: pd.DataFrame) -> dict:
    result = {
        'stats':  _compute_stats(df),
        'charts': {
            'target_dist':    _target_dist(df),
            'missing':        _missing(df),
            'experience':     _experience(df),
            'education':      _education(df),
            'company':        _company(df),
            'demographics':   _demographics(df),
            'training_hours': _training_hours(df),
            'cdi':            _cdi(df),
            'correlation':    _correlation(df),
            'feature_target': _feature_target(df),
        }
    }
    return result


def _compute_stats(df):
    return {
        'total_candidates':    int(len(df)),
        'seekers':             int((df['target'] == 1).sum()),
        'stayers':             int((df['target'] == 0).sum()),
        'seek_rate':           round(float((df['target'] == 1).mean() * 100), 1),
        'missing_cells':       int(df.isnull().sum().sum()),
        'missing_pct':         round(float(df.isnull().mean().mean() * 100), 2),
        'duplicates':          int(df.duplicated().sum()),
        'features':            int(len(df.columns) - 1),
        'numeric_features':    int(len(df.select_dtypes(include=np.number).columns)),
        'categorical_features':int(len(df.select_dtypes(include='object').columns)),
        'avg_training_hours':  round(float(df['training_hours'].mean()), 1),
        'median_training_hours': float(df['training_hours'].median()),
        'avg_cdi':             round(float(df['city_development_index'].mean()), 3),
    }


def _target_dist(df):
    vc = df['target'].value_counts().sort_index()
    labels = ['Staying', 'Job Seeking']
    values = [int(vc.get(0.0, 0)), int(vc.get(1.0, 0))]

    return _j({
        'data': [
            {'type': 'bar', 'x': labels, 'y': values,
             'marker': {'color': [ACCENT, VIOLET], 'line': {'width': 0}},
             'text': [f'{v:,}' for v in values], 'textposition': 'outside',
             'textfont': {'color': TEXT, 'size': 12},
             'hovertemplate': '%{x}: %{y:,}<extra></extra>'},
        ],
        'layout': _layout('Target Distribution — Stay vs. Job Seeking', height=360,
                           yaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
                           showlegend=False)
    })


def _missing(df):
    miss = (df.isnull().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values(ascending=True)
    if len(miss) == 0:
        return _j({'data': [{'type': 'bar', 'x': [0], 'y': ['No missing values'],
                              'marker': {'color': EMERALD}}],
                   'layout': _layout('Missing Values — None Found', height=300)})
    colors = [ROSE if v > 30 else AMBER if v > 10 else ACCENT for v in miss.values]
    return _j({
        'data': [{'type': 'bar', 'orientation': 'h',
                  'x': miss.values.tolist(), 'y': miss.index.tolist(),
                  'marker': {'color': colors},
                  'text': [f'{v:.1f}%' for v in miss.values], 'textposition': 'outside',
                  'hovertemplate': '%{y}: %{x:.1f}%<extra></extra>'}],
        'layout': _layout('Missing Values by Feature', height=max(300, len(miss)*40),
                           xaxis=dict(title='Missing %', gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
                           shapes=[{'type':'line','x0':30,'x1':30,'y0':-0.5,'y1':len(miss)-0.5,
                                    'line':{'color':ROSE,'width':1,'dash':'dot'}}],
                           annotations=[{'x':30,'y':len(miss)-0.5,'text':'High (30%)',
                                         'showarrow':False,'font':{'color':ROSE,'size':10}}])
    })


def _experience(df):
    exp_order = [str(i) for i in range(1, 21)] + ['>20', '<1']
    ec = df['experience'].value_counts().reindex(exp_order).fillna(0)

    seek = df.groupby('experience')['target'].mean().reindex(exp_order).fillna(0) * 100

    return _j({
        'data': [
            {'type': 'bar', 'x': exp_order, 'y': ec.values.tolist(),
             'name': 'Count', 'marker': {'color': ACCENT, 'opacity': 0.85},
             'hovertemplate': '%{x} yrs: %{y:,}<extra></extra>'},
            {'type': 'scatter', 'x': exp_order, 'y': seek.values.tolist(),
             'name': 'Seek Rate %', 'yaxis': 'y2', 'mode': 'lines+markers',
             'line': {'color': VIOLET, 'width': 2},
             'marker': {'size': 5},
             'hovertemplate': '%{x} yrs seek rate: %{y:.1f}%<extra></extra>'},
        ],
        'layout': _layout('Experience Distribution + Seek Rate', height=380,
                           yaxis2=dict(overlaying='y', side='right', tickfont=dict(color=VIOLET),
                                       gridcolor='rgba(0,0,0,0)', title=dict(text='Seek Rate %', font=dict(color=VIOLET))),
                           legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=10)))
    })


def _education(df):
    edu = df.groupby('education_level').agg(
        count=('target', 'count'),
        seek_rate=('target', 'mean')
    ).reset_index()
    edu['seek_rate'] = (edu['seek_rate'] * 100).round(1)
    edu = edu.sort_values('count', ascending=False)

    return _j({
        'data': [
            {'type': 'bar', 'x': edu['education_level'].tolist(), 'y': edu['count'].tolist(),
             'name': 'Count', 'marker': {'color': PALETTE[:len(edu)], 'opacity': 0.9},
             'text': edu['count'].tolist(), 'textposition': 'outside',
             'hovertemplate': '%{x}: %{y:,}<extra></extra>'},
            {'type': 'scatter', 'x': edu['education_level'].tolist(), 'y': edu['seek_rate'].tolist(),
             'name': 'Seek Rate %', 'yaxis': 'y2', 'mode': 'lines+markers',
             'line': {'color': ROSE, 'width': 2}, 'marker': {'size': 7},
             'hovertemplate': 'Seek rate: %{y:.1f}%<extra></extra>'},
        ],
        'layout': _layout('Education Level vs. Job-Seeking Rate', height=380,
                           yaxis2=dict(overlaying='y', side='right', tickfont=dict(color=ROSE),
                                       gridcolor='rgba(0,0,0,0)'))
    })


def _company(df):
    size_order = ['<10','10-49','50-99','100-500','500-999','1000-4999','5000-9999','10000+']
    cs = df['company_size'].value_counts().reindex(size_order).fillna(0)
    cs_seek = df.groupby('company_size')['target'].mean().reindex(size_order).fillna(0) * 100

    ct = df['company_type'].value_counts()
    ct_seek = df.groupby('company_type')['target'].mean() * 100

    return _j({
        'data': [
            {'type': 'bar', 'x': size_order, 'y': cs.values.tolist(),
             'name': 'Count by Size', 'marker': {'color': ACCENT, 'opacity': 0.85},
             'xaxis': 'x', 'yaxis': 'y',
             'hovertemplate': '%{x}: %{y:,}<extra></extra>'},
            {'type': 'scatter', 'x': size_order, 'y': cs_seek.values.tolist(),
             'name': 'Size Seek %', 'mode': 'lines+markers',
             'line': {'color': VIOLET, 'width': 2}, 'marker': {'size': 6},
             'xaxis': 'x', 'yaxis': 'y2',
             'hovertemplate': 'Seek rate: %{y:.1f}%<extra></extra>'},
            {'type': 'bar', 'x': ct.index.tolist(), 'y': ct_seek.values.tolist(),
             'name': 'Type Seek %', 'marker': {'color': PALETTE[:len(ct)], 'opacity': 0.9},
             'xaxis': 'x3', 'yaxis': 'y3',
             'hovertemplate': '%{x}: %{y:.1f}%<extra></extra>'},
        ],
        'layout': dict(
            paper_bgcolor=PANEL, plot_bgcolor=PANEL2,
            font=dict(family='DM Sans', color=MUTED, size=11),
            height=420, margin=dict(l=48, r=48, t=56, b=80),
            title=dict(text='Company Profile Analysis', font=dict(family='Syne', size=15, color=TEXT), x=0.02),
            grid=dict(rows=1, columns=2, pattern='independent'),
            xaxis=dict(domain=[0, 0.48], gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED), tickangle=30),
            yaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
            yaxis2=dict(overlaying='y', side='right', tickfont=dict(color=VIOLET), gridcolor='rgba(0,0,0,0)'),
            xaxis3=dict(domain=[0.52, 1.0], gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED), tickangle=15, anchor='y3'),
            yaxis3=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED), anchor='x3'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=10)),
        )
    })


def _demographics(df):
    g_seek = df.groupby('gender')['target'].mean() * 100
    univ_seek = df.groupby('enrolled_university')['target'].mean() * 100

    return _j({
        'data': [
            {'type': 'bar', 'x': g_seek.index.tolist(), 'y': g_seek.values.tolist(),
             'name': 'By Gender', 'marker': {'color': PALETTE[:len(g_seek)]},
             'xaxis': 'x', 'yaxis': 'y',
             'text': [f'{v:.1f}%' for v in g_seek.values], 'textposition': 'outside',
             'hovertemplate': '%{x}: %{y:.1f}%<extra></extra>'},
            {'type': 'bar', 'x': univ_seek.index.tolist(), 'y': univ_seek.values.tolist(),
             'name': 'By University', 'marker': {'color': PALETTE[3:3+len(univ_seek)]},
             'xaxis': 'x2', 'yaxis': 'y2',
             'text': [f'{v:.1f}%' for v in univ_seek.values], 'textposition': 'outside',
             'hovertemplate': '%{x}: %{y:.1f}%<extra></extra>'},
        ],
        'layout': dict(
            paper_bgcolor=PANEL, plot_bgcolor=PANEL2,
            font=dict(family='DM Sans', color=MUTED, size=11),
            height=380, margin=dict(l=48, r=48, t=56, b=80),
            title=dict(text='Job-Seeking Rate by Demographics', font=dict(family='Syne', size=15, color=TEXT), x=0.02),
            grid=dict(rows=1, columns=2, pattern='independent'),
            xaxis=dict(domain=[0, 0.45], gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
            yaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED), title=dict(text='Seek Rate %', font=dict(color=MUTED))),
            xaxis2=dict(domain=[0.55, 1.0], gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED), anchor='y2', tickangle=15),
            yaxis2=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED), anchor='x2'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=10)),
            showlegend=False,
        )
    })


def _training_hours(df):
    seekers = df[df['target'] == 1]['training_hours'].tolist()
    stayers = df[df['target'] == 0]['training_hours'].tolist()

    return _j({
        'data': [
            {'type': 'histogram', 'x': stayers, 'name': 'Staying',
             'marker': {'color': ACCENT, 'opacity': 0.7}, 'nbinsx': 40,
             'hovertemplate': 'Hours: %{x}<br>Count: %{y}<extra></extra>'},
            {'type': 'histogram', 'x': seekers, 'name': 'Job Seeking',
             'marker': {'color': VIOLET, 'opacity': 0.7}, 'nbinsx': 40,
             'hovertemplate': 'Hours: %{x}<br>Count: %{y}<extra></extra>'},
        ],
        'layout': _layout('Training Hours Distribution by Target', height=380,
                           barmode='overlay',
                           xaxis=dict(title='Training Hours', gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
                           yaxis=dict(title='Count', gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)))
    })


def _cdi(df):
    bins = pd.cut(df['city_development_index'], bins=30)
    df2 = df.copy()
    df2['cdi_bin'] = bins

    seekers = df[df['target'] == 1]['city_development_index'].tolist()
    stayers = df[df['target'] == 0]['city_development_index'].tolist()

    # Scatter: CDI vs training hours
    sample = df.sample(min(2000, len(df)), random_state=42)

    return _j({
        'data': [
            {'type': 'histogram', 'x': stayers, 'name': 'Staying',
             'marker': {'color': ACCENT, 'opacity': 0.7}, 'nbinsx': 35,
             'hovertemplate': 'CDI: %{x:.3f}<br>Count: %{y}<extra></extra>'},
            {'type': 'histogram', 'x': seekers, 'name': 'Job Seeking',
             'marker': {'color': VIOLET, 'opacity': 0.7}, 'nbinsx': 35,
             'hovertemplate': 'CDI: %{x:.3f}<br>Count: %{y}<extra></extra>'},
        ],
        'layout': _layout('City Development Index Distribution by Target', height=380,
                           barmode='overlay',
                           xaxis=dict(title='City Development Index', gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
                           yaxis=dict(title='Count', gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)))
    })


def _correlation(df):
    num_df = df.select_dtypes(include=np.number).drop(columns=['enrollee_id'], errors='ignore')
    corr = num_df.corr().round(3)
    cols = corr.columns.tolist()

    # mask upper triangle
    z = corr.values.tolist()
    for i in range(len(z)):
        for j in range(i+1, len(z[i])):
            z[i][j] = None

    return _j({
        'data': [{
            'type': 'heatmap',
            'z': z, 'x': cols, 'y': cols,
            'colorscale': [[0,'#2563EB'],[0.5,'#1C2640'],[1,'#F43F5E']],
            'zmid': 0,
            'text': [[f'{v:.2f}' if v is not None else '' for v in row] for row in z],
            'texttemplate': '%{text}',
            'textfont': {'size': 9, 'color': TEXT},
            'hovertemplate': '%{y} vs %{x}: %{z:.3f}<extra></extra>',
            'showscale': True,
            'colorbar': {'tickfont': {'color': MUTED}, 'bgcolor': PANEL2},
        }],
        'layout': _layout('Feature Correlation Matrix', height=420,
                           xaxis=dict(tickangle=30, gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED, size=9)),
                           yaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED, size=9)))
    })


def _feature_target(df):
    features = ['relevent_experience', 'enrolled_university', 'education_level',
                'major_discipline', 'last_new_job']

    x_all, y_all, text_all, color_all = [], [], [], []
    for feat in features:
        if feat not in df.columns:
            continue
        rates = (df.groupby(feat)['target'].mean() * 100).sort_values(ascending=False)
        for val, rate in rates.items():
            x_all.append(f'{feat[:10]}: {str(val)[:12]}')
            y_all.append(round(float(rate), 1))
            text_all.append(f'{rate:.1f}%')
            color_all.append(VIOLET if rate > 25 else ACCENT)

    baseline = float(df['target'].mean() * 100)

    return _j({
        'data': [
            {'type': 'bar', 'x': x_all, 'y': y_all,
             'marker': {'color': color_all},
             'text': text_all, 'textposition': 'outside',
             'textfont': {'size': 8},
             'hovertemplate': '%{x}<br>Seek rate: %{y:.1f}%<extra></extra>'},
        ],
        'layout': _layout('Feature vs. Job-Seek Rate (above baseline = purple)', height=420,
                           xaxis=dict(tickangle=45, tickfont=dict(color=MUTED, size=8), gridcolor=RIM, linecolor=RIM),
                           yaxis=dict(title='Seek Rate %', gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
                           shapes=[{'type':'line','x0':-0.5,'x1':len(x_all)-0.5,
                                    'y0':baseline,'y1':baseline,
                                    'line':{'color':AMBER,'width':1.5,'dash':'dot'}}],
                           annotations=[{'x':len(x_all)-1,'y':baseline+0.5,
                                         'text':f'Baseline {baseline:.1f}%',
                                         'showarrow':False,'font':{'color':AMBER,'size':9}}])
    })
