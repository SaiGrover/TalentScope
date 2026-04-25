"""
Model Trainer — returns Plotly JSON charts + full feature interpretability
"""
import numpy as np
import json, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     roc_curve, precision_recall_curve,
                                     average_precision_score)
from sklearn.inspection      import permutation_importance

DEFAULT_THRESHOLD = 0.45
PERM_IMPORTANCE_MAX_ROWS = 500
PERM_IMPORTANCE_N_REPEATS = 6

# ── Plotly theme ──────────────────────────────────────────────────────────────
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
PALETTE = [ACCENT, VIOLET, EMERALD, AMBER, ROSE, CYAN]

MODEL_COLORS = {
    'Logistic Regression': ACCENT,
    'Decision Tree':       VIOLET,
    'Random Forest':       EMERALD,
    'SVM':                 ROSE,
    'KNN':                 AMBER,
}

def _layout(title='', height=380, **extra):
    base = dict(
        title=dict(text=title, font=dict(family='Syne,sans-serif', size=14, color=TEXT), x=0.02),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL2,
        font=dict(family='DM Sans,sans-serif', color=MUTED, size=11),
        height=height,
        margin=dict(l=54, r=24, t=54, b=54),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=MUTED, size=10)),
        xaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
    )
    base.update(extra)
    return base

def _j(d):
    return json.loads(json.dumps(d, default=lambda x: float(x) if isinstance(x,(np.floating,np.integer)) else str(x)))

def _best_threshold(y_true, y_proba):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), round(best_f1, 4)


class ModelTrainer:
    MODEL_CONFIGS = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
        'params': {
            'C': [0.01,0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'max_depth': [5, 10,20],
            'min_samples_split': [2,10, 20],
            'min_samples_leaf' : [1,5,10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, None]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42, class_weight='balanced'),
        'params': {
            'C': [0.5, 1],
            'kernel': ['rbf']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [5, 7, 9]
        }
    }
}
    SLOW_MODELS  = {'SVM', 'KNN'}
    MAX_SAMPLES  = 3000

    def __init__(self):
        self.models      = {}
        self.results     = {}
        self.feature_names = []
        self.X_val = self.y_val = None

    def train_all(self, X, y):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        self.X_val, self.y_val = X_val, y_val
        self.feature_names = self.feature_names or []

        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rng = np.random.RandomState(42)
        if len(y_tr) > self.MAX_SAMPLES:
            idx = rng.choice(len(y_tr), self.MAX_SAMPLES, replace=False)
            X_sm, y_sm = X_tr[idx], y_tr[idx]
        else:
            X_sm, y_sm = X_tr, y_tr

        for name, config in self.MODEL_CONFIGS.items():
            base_model = config['model']
            param_grid = config['params']

            Xt, yt = (X_sm, y_sm) if name in self.SLOW_MODELS else (X_tr, y_tr)

            grid = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=1
            )

            grid.fit(Xt, yt)

            clf = grid.best_estimator_
            self.models[name] = clf

            y_pv  = clf.predict(X_val)
            y_pb  = clf.predict_proba(X_val)[:,1] if hasattr(clf,'predict_proba') else None
            y_ptr = clf.predict(Xt)

            opt_t, opt_f1 = (0.5, f1_score(y_val,y_pv,zero_division=0))
            default_t = DEFAULT_THRESHOLD
            if y_pb is not None:
                opt_t, opt_f1 = _best_threshold(y_val, y_pb)
                y_tuned = (y_pb >= opt_t).astype(int)
                y_fixed = (y_pb >= default_t).astype(int)
            else:
                y_tuned = y_pv
                y_fixed = y_pv

            cv_s = grid.cv_results_['mean_test_score']
            cv_mean = float(np.max(cv_s))
            cv_std = float(np.std(cv_s))
            train_f1 = float(f1_score(yt, y_ptr, zero_division=0))
            val_f1   = float(f1_score(y_val, y_pv, zero_division=0))
            try:
                roc = roc_auc_score(y_val, y_pb) if y_pb is not None else 0
            except:
                roc = 0
            m = {
                'accuracy':    round(float(accuracy_score(y_val, y_pv)), 4),
                'precision':   round(float(precision_score(y_val, y_pv, zero_division=0)), 4),
                'recall':      round(float(recall_score(y_val, y_pv, zero_division=0)), 4),
                'f1':          round(float(f1_score(y_val, y_pv, zero_division=0)), 4),
                'roc_auc': round(float(roc), 4),
                'opt_threshold':   opt_t,
                'default_threshold': default_t,
                'f1_tuned':        round(float(f1_score(y_val, y_tuned, zero_division=0)), 4),
                'precision_tuned': round(float(precision_score(y_val, y_tuned, zero_division=0)), 4),
                'recall_tuned':    round(float(recall_score(y_val, y_tuned, zero_division=0)), 4),
                'f1_fixed':        round(float(f1_score(y_val, y_fixed, zero_division=0)), 4),
                'precision_fixed': round(float(precision_score(y_val, y_fixed, zero_division=0)), 4),
                'recall_fixed':    round(float(recall_score(y_val, y_fixed, zero_division=0)), 4),
                'cv_f1_mean': round(cv_mean, 4),
                'cv_f1_std': round(cv_std, 4),
                'train_f1':    round(train_f1, 4),
                'val_f1':      round(val_f1,   4),
                'overfit_flag': bool(train_f1 - val_f1 > 0.15),
                'confusion_matrix': confusion_matrix(y_val, y_pv).tolist(),
                'best_params': grid.best_params_,
            }
            if y_pb is not None:
                fpr,tpr,_ = roc_curve(y_val, y_pb)
                m['roc_fpr'] = fpr.tolist(); m['roc_tpr'] = tpr.tolist()
                pr_p,pr_r,_ = precision_recall_curve(y_val, y_pb)
                m['pr_prec'] = pr_p.tolist(); m['pr_rec'] = pr_r.tolist()
                m['avg_precision'] = round(float(average_precision_score(y_val, y_pb)), 4)
            self.results[name] = m

        # Feature importances
        for name in ('Random Forest','Decision Tree'):
            if name in self.models and hasattr(self.models[name],'feature_importances_'):
                self.results[name]['feature_importances'] = self.models[name].feature_importances_.tolist()

        # Permutation importance for LR (model-agnostic)
        lr = self.models.get('Logistic Regression')
        if lr is not None:
            try:
                pi = permutation_importance(lr, X_val, y_val, n_repeats=5,
                                            random_state=42, scoring='roc_auc', n_jobs=1)
                self.results['Logistic Regression']['perm_importances_mean'] = pi.importances_mean.tolist()
                self.results['Logistic Regression']['perm_importances_std']  = pi.importances_std.tolist()
            except Exception:
                pass

        # Build LR coefficients for interpretability
        lr_coef = None
        if lr is not None and hasattr(lr, 'coef_'):
            lr_coef = lr.coef_[0].tolist()

        # Charts
        self.results['_charts'] = {
            'performance':       self._chart_performance(),
            'roc_curves':        self._chart_roc(),
            'pr_curves':         self._chart_pr(),
            'confusion':         self._chart_confusion(),
            'radar':             self._chart_radar(),
            'feature_importance':self._chart_feature_importance(),
            'lr_coefficients':   self._chart_lr_coef(lr_coef),
            'overfit_diagnostic':self._chart_overfit(),
            'threshold_tuning':  self._chart_threshold(),
            'cv_scores':         self._chart_cv(),
            'shap_surrogate':    self._chart_shap_surrogate(X_val, y_val),
        }

        best = max((k for k in self.results if not k.startswith('_')),
                   key=lambda k: self.results[k]['roc_auc'])
        self.results['_meta'] = {
            'best_model':        best,
            'val_size':          int(len(y_val)),
            'train_size':        int(len(y_tr)),
            'val_positive_rate': round(float(y_val.mean()), 3),
        }

        # Feature interpretability summary
        self.results['_interpretability'] = self._build_interpretability(X_val, y_val)

        return self.results

    # ── Charts ─────────────────────────────────────────────────────────────────

    def _chart_performance(self):
        metrics = ['accuracy','precision','recall','f1','roc_auc']
        mlbls   = ['Accuracy','Precision','Recall','F1','ROC-AUC']
        names   = list(self.MODEL_CONFIGS.keys())
        traces  = []
        for i, name in enumerate(names):
            r = self.results[name]
            traces.append({
                'type': 'bar', 'name': name,
                'x': mlbls,
                'y': [r[m] for m in metrics],
                'marker': {'color': list(MODEL_COLORS.values())[i], 'opacity': 0.87},
                'hovertemplate': f'{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>',
            })
        return _j({'data': traces, 'layout': _layout('Model Performance — Validation Set', height=400,
                   barmode='group', yaxis=dict(range=[0,1.12], gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED)),
                   shapes=[{'type':'line','x0':-0.5,'x1':4.5,'y0':0.5,'y1':0.5,'line':{'color':MUTED,'width':1,'dash':'dot'}}])})

    def _chart_roc(self):
        traces = []
        traces.append({'type':'scatter','x':[0,1],'y':[0,1],'mode':'lines',
                       'line':{'color':MUTED,'width':1,'dash':'dash'},'name':'Random','showlegend':False})
        for name, color in MODEL_COLORS.items():
            r = self.results[name]
            if 'roc_fpr' in r:
                traces.append({'type':'scatter','name':f"{name} AUC={r['roc_auc']:.3f}",
                                'x':r['roc_fpr'],'y':r['roc_tpr'],'mode':'lines',
                                'line':{'color':color,'width':2},
                                'hovertemplate':f'{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'})
        return _j({'data':traces,'layout':_layout('ROC Curves — Validation Set', height=400,
                   xaxis=dict(title='False Positive Rate',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)),
                   yaxis=dict(title='True Positive Rate',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)))})

    def _chart_pr(self):
        baseline = float(self.y_val.mean()) if self.y_val is not None else 0.25
        traces = [{'type':'scatter','x':[0,1],'y':[baseline,baseline],'mode':'lines',
                   'line':{'color':MUTED,'width':1,'dash':'dash'},'name':f'Baseline {baseline:.2f}','showlegend':True}]
        for name, color in MODEL_COLORS.items():
            r = self.results[name]
            if 'pr_prec' in r:
                traces.append({'type':'scatter','name':f"{name} AP={r.get('avg_precision',0):.3f}",
                                'x':r['pr_rec'],'y':r['pr_prec'],'mode':'lines',
                                'line':{'color':color,'width':2},
                                'hovertemplate':f'{name}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>'})
        return _j({'data':traces,'layout':_layout('Precision-Recall Curves — Validation Set', height=400,
                   xaxis=dict(title='Recall',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)),
                   yaxis=dict(title='Precision',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)))})

    def _chart_confusion(self):
        names = list(self.MODEL_CONFIGS.keys())
        cols = min(5, len(names))
        traces = []
        annotations = []
        for i, name in enumerate(names):
            cm = np.array(self.results[name]['confusion_matrix'])
            # normalise by row
            cm_n = cm.astype(float)
            for r in range(2): cm_n[r] = cm_n[r] / cm_n[r].sum() if cm_n[r].sum() else cm_n[r]
            domain_x = [i/cols + 0.01, (i+1)/cols - 0.01]
            traces.append({'type':'heatmap','z':cm_n.tolist(),
                            'x':['Pred Stay','Pred Leave'],'y':['True Stay','True Leave'],
                            'xaxis':f'x{i+1}' if i else 'x','yaxis':f'y{i+1}' if i else 'y',
                            'colorscale':[[0,PANEL2],[1,ACCENT]],'showscale':False,
                            'hovertemplate':f'{name}<br>%{{y}} → %{{x}}: %{{z:.2f}}<extra></extra>'})
            for r in range(2):
                for c in range(2):
                    annotations.append({'x':['Pred Stay','Pred Leave'][c],'y':['True Stay','True Leave'][r],
                                        'text':f'{cm[r,c]:,}<br>{cm_n[r,c]:.0%}','showarrow':False,
                                        'font':{'color':TEXT,'size':9},
                                        'xref':f'x{i+1}' if i else 'x',
                                        'yref':f'y{i+1}' if i else 'y'})
            annotations.append({'x':sum(domain_x)/2,'y':1.08,'xref':'paper','yref':'paper',
                                 'text':f'<b>{name}</b>','showarrow':False,'font':{'color':TEXT,'size':10}})

        axes = {}
        for i in range(cols):
            domain_x = [i/cols+0.01,(i+1)/cols-0.01]
            suf = str(i+1) if i else ''
            axes[f'xaxis{suf}'] = dict(domain=domain_x, gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED,size=8))
            axes[f'yaxis{suf}'] = dict(gridcolor=RIM, linecolor=RIM, tickfont=dict(color=MUTED,size=8), anchor=f'x{i+1}' if i else 'x')
        layout = dict(paper_bgcolor=PANEL, plot_bgcolor=PANEL2,
                      font=dict(family='DM Sans',color=MUTED,size=10),
                      height=280, margin=dict(l=40,r=10,t=60,b=40),
                      title=dict(text='Confusion Matrices (row-normalised) — Validation Set',
                                 font=dict(family='Syne',size=14,color=TEXT),x=0.02),
                      annotations=annotations, showlegend=False)
        layout.update(axes)
        return _j({'data':traces,'layout':layout})

    def _chart_radar(self):
        metrics = ['accuracy','precision','recall','f1','roc_auc']
        mlbls   = ['Accuracy','Precision','Recall','F1','ROC-AUC']
        traces  = []
        for name, color in MODEL_COLORS.items():
            r   = self.results[name]
            vals = [r[m] for m in metrics] + [r[metrics[0]]]
            lbls = mlbls + [mlbls[0]]
            traces.append({'type':'scatterpolar','r':vals,'theta':lbls,'fill':'toself',
                            'name':name,'line':{'color':color,'width':2},
                            'fillcolor':color.replace('#','rgba(').rstrip(')')+',0.07)',
                            'marker':{'size':4},
                            'hovertemplate':f'{name}<br>%{{theta}}: %{{r:.3f}}<extra></extra>'})
        return _j({'data':traces,'layout':dict(
            paper_bgcolor=PANEL, plot_bgcolor=PANEL2,
            font=dict(family='DM Sans',color=MUTED,size=10), height=440,
            margin=dict(l=40,r=120,t=60,b=40),
            title=dict(text='Radar — Multi-Metric Comparison (Validation)',font=dict(family='Syne',size=14,color=TEXT),x=0.02),
            polar=dict(bgcolor=PANEL2,
                       radialaxis=dict(visible=True,range=[0,1],gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED,size=8)),
                       angularaxis=dict(gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED,size=10))),
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color=MUTED,size=10),x=1.02))})

    def _chart_feature_importance(self):
        rf = self.models.get('Random Forest')
        if rf is None or not hasattr(rf,'feature_importances_'):
            return None
        fi    = rf.feature_importances_
        names = self.feature_names if self.feature_names and len(self.feature_names)==len(fi) else [f'F{i}' for i in range(len(fi))]
        pairs = sorted(zip(names, fi), key=lambda x: x[1])
        top   = pairs[-16:]
        ns, vs = zip(*top)
        colors = [EMERALD if v == max(vs) else ACCENT for v in vs]
        return _j({'data':[{'type':'bar','orientation':'h',
                             'x':list(vs),'y':list(ns),
                             'marker':{'color':colors},
                             'text':[f'{v:.3f}' for v in vs],'textposition':'outside',
                             'hovertemplate':'%{y}: %{x:.4f}<extra></extra>'}],
                   'layout':_layout('Random Forest — Feature Importances (Training Split)', height=480,
                                    xaxis=dict(title='Importance',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)),
                                    yaxis=dict(gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED,size=10)),
                                    margin=dict(l=160,r=60,t=54,b=40))})

    def _chart_lr_coef(self, coef):
        if coef is None or not self.feature_names:
            return None
        names = self.feature_names if len(self.feature_names)==len(coef) else [f'F{i}' for i in range(len(coef))]
        pairs = sorted(zip(names, coef), key=lambda x: x[1])
        ns, vs = zip(*pairs)
        colors = [ROSE if v > 0 else ACCENT for v in vs]
        return _j({'data':[{'type':'bar','orientation':'h',
                             'x':list(vs),'y':list(ns),
                             'marker':{'color':colors},
                             'text':[f'{v:+.3f}' for v in vs],'textposition':'outside',
                             'hovertemplate':'%{y}: %{x:+.3f}<extra></extra>'}],
                   'layout':_layout('Logistic Regression — Coefficients (positive = pushes toward job-seeking)', height=480,
                                    xaxis=dict(title='Coefficient',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED),zeroline=True,zerolinecolor=MUTED,zerolinewidth=1),
                                    yaxis=dict(gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED,size=10)),
                                    margin=dict(l=160,r=60,t=54,b=40),
                                    shapes=[{'type':'line','x0':0,'x1':0,'y0':-0.5,'y1':len(ns)-0.5,
                                             'line':{'color':MUTED,'width':1,'dash':'dot'}}])})

    def _chart_overfit(self):
        names = list(self.MODEL_CONFIGS.keys())
        tf = [self.results[n]['train_f1'] for n in names]
        vf = [self.results[n]['val_f1']   for n in names]
        of = [self.results[n]['overfit_flag'] for n in names]
        annotations = []
        for i,(t,v,o) in enumerate(zip(tf,vf,of)):
            if o:
                annotations.append({'x':names[i],'y':max(t,v)+0.02,'text':f'⚠ gap={t-v:.2f}',
                                     'showarrow':False,'font':{'color':ROSE,'size':9}})
        return _j({'data':[
            {'type':'bar','name':'Train F1','x':names,'y':tf,'marker':{'color':ACCENT,'opacity':0.85},
             'hovertemplate':'Train F1: %{y:.3f}<extra></extra>'},
            {'type':'bar','name':'Val F1','x':names,'y':vf,'marker':{'color':EMERALD,'opacity':0.85},
             'hovertemplate':'Val F1: %{y:.3f}<extra></extra>'},
        ],'layout':_layout('Overfitting Diagnostic — Train vs. Validation F1', height=380,
                            barmode='group',yaxis=dict(range=[0,1.05],gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)),
                            annotations=annotations)})

    def _chart_threshold(self):
        names = list(self.MODEL_CONFIGS.keys())
        fd  = [self.results[n]['f1']       for n in names]
        ft  = [self.results[n]['f1_tuned'] for n in names]
        thr = [self.results[n]['opt_threshold'] for n in names]
        annotations = [{'x':n,'y':ft[i]+0.01,'text':f't={thr[i]}','showarrow':False,
                         'font':{'color':MUTED,'size':9}} for i,n in enumerate(names)]
        return _j({'data':[
            {'type':'bar','name':'F1 @ 0.50','x':names,'y':fd,'marker':{'color':VIOLET,'opacity':0.85},
             'hovertemplate':'F1 @0.5: %{y:.3f}<extra></extra>'},
            {'type':'bar','name':'F1 @ optimal t','x':names,'y':ft,'marker':{'color':EMERALD,'opacity':0.85},
             'hovertemplate':'F1 tuned: %{y:.3f}<extra></extra>'},
        ],'layout':_layout('Threshold Tuning — F1 Improvement per Model', height=380,
                            barmode='group',yaxis=dict(range=[0,1.0],gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)),
                            annotations=annotations)})

    def _chart_cv(self):
        names  = list(self.MODEL_CONFIGS.keys())
        means  = [self.results[n]['cv_f1_mean'] for n in names]
        stds   = [self.results[n]['cv_f1_std']  for n in names]
        colors = list(MODEL_COLORS.values())
        return _j({'data':[{'type':'bar','x':names,'y':means,
                             'error_y':{'type':'data','array':stds,'visible':True,
                                        'color':TEXT,'thickness':1.5,'width':6},
                             'marker':{'color':colors,'opacity':0.87},
                             'text':[f'{m:.3f}' for m in means],'textposition':'outside',
                             'hovertemplate':'%{x}<br>CV F1: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>'}],
                   'layout':_layout('5-Fold CV F1 Score ± Std Dev', height=380,
                                    yaxis=dict(range=[0,max(means)+0.12],gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)))})

    def _chart_shap_surrogate(self, X_val, y_val):
        """
        SHAP-surrogate: use RF feature importances as a proxy and show
        a population-level beeswarm approximation using permutation importance.
        """
        rf = self.models.get('Random Forest')
        if rf is None or X_val is None:
            return None
        names = self.feature_names if self.feature_names and len(self.feature_names)==X_val.shape[1] else [f'F{i}' for i in range(X_val.shape[1])]
        X_perm, y_perm = X_val, y_val
        if X_val.shape[0] > PERM_IMPORTANCE_MAX_ROWS:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_val.shape[0], PERM_IMPORTANCE_MAX_ROWS, replace=False)
            X_perm, y_perm = X_val[idx], y_val[idx]
        try:
            pi = permutation_importance(rf, X_perm, y_perm, n_repeats=PERM_IMPORTANCE_N_REPEATS,
                                        random_state=42, scoring='roc_auc', n_jobs=1)
            imp_mean = pi.importances_mean
            imp_std  = pi.importances_std
        except Exception:
            return None

        pairs = sorted(zip(names, imp_mean, imp_std), key=lambda x: x[1])
        top = pairs[-14:]
        ns, vs, ss = zip(*top)
        colors = [EMERALD if v>0.02 else ACCENT if v>0 else ROSE for v in vs]
        return _j({'data':[{'type':'bar','orientation':'h',
                             'x':list(vs),'y':list(ns),
                             'error_x':{'type':'data','array':list(ss),'visible':True,'color':MUTED,'thickness':1.5,'width':4},
                             'marker':{'color':colors},
                             'text':[f'{v:.4f}' for v in vs],'textposition':'outside',
                             'hovertemplate':'%{y}<br>Perm importance: %{x:.4f}<extra></extra>'}],
                   'layout':_layout('Permutation Importance — RF on Validation Set<br><sup>Drop in ROC-AUC when feature shuffled. Larger = more important.</sup>',
                                    height=480,
                                    xaxis=dict(title='Drop in ROC-AUC',gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED)),
                                    yaxis=dict(gridcolor=RIM,linecolor=RIM,tickfont=dict(color=MUTED,size=10)),
                                    margin=dict(l=160,r=70,t=70,b=40))})

    # ── Interpretability summary ───────────────────────────────────────────────

    def _build_interpretability(self, X_val, y_val):
        """
        Build human-readable interpretability block:
        - top features by RF importance
        - LR coefficient direction labels
        - decision path description
        """
        rf = self.models.get('Random Forest')
        lr = self.models.get('Logistic Regression')
        names = self.feature_names

        result = {'top_features': [], 'lr_directions': [], 'key_rules': []}

        # Top features
        if rf and hasattr(rf,'feature_importances_') and names:
            fi = rf.feature_importances_
            pairs = sorted(zip(names,fi), key=lambda x:-x[1])[:6]
            READABLE = {
                'city_development_index': 'City Development Index',
                'cdi_x_rel_exp':          'CDI × Relevant Experience',
                'company_size_num':        'Company Size',
                'company_type':            'Company Type',
                'training_hours':          'Training Hours',
                'experience_num':          'Years of Experience',
                'last_new_job_num':        'Time Since Last Job Change',
                'major_discipline':        'Major / Discipline',
                'education_level':         'Education Level',
                'high_cdi':                'High CDI City Flag',
                'has_rel_exp':             'Has Relevant Experience',
                'is_enrolled':             'Currently Enrolled',
            }
            DIRECTION = {
                'city_development_index': 'Low CDI → higher seek probability (candidates in less developed cities seek better opportunities)',
                'cdi_x_rel_exp':          'Low CDI + no relevant experience → strongest seeker signal',
                'company_size_num':        'Smaller company → higher likelihood of seeking',
                'company_type':            '"Other" and "Early Stage Startup" employees seek more',
                'training_hours':          'More training hours → slightly higher seek probability',
                'experience_num':          'Less experience (<5 yrs) → more likely to switch',
                'last_new_job_num':        'Changed job recently → less likely to seek again',
                'major_discipline':        'STEM disciplines seek more than Arts/Humanities',
                'education_level':         'Graduate level seeks most; PhD seeks least',
                'high_cdi':                'Low-CDI cities drive job-seeking behavior',
                'has_rel_exp':             'Candidates without relevant experience seek more',
                'is_enrolled':             'Full-time enrolled candidates more likely to seek',
            }
            for name, imp in pairs:
                result['top_features'].append({
                    'feature':   READABLE.get(name, name),
                    'raw_name':  name,
                    'importance': round(float(imp), 4),
                    'pct':       round(float(imp)*100, 1),
                    'direction': DIRECTION.get(name, ''),
                })

        # LR coefficients direction
        if lr and hasattr(lr,'coef_') and names:
            coef = lr.coef_[0]
            pairs = sorted(zip(names,coef), key=lambda x:-abs(x[1]))[:5]
            for name, c in pairs:
                result['lr_directions'].append({
                    'feature': name,
                    'coef':    round(float(c), 4),
                    'push':    'job-seeking' if c > 0 else 'staying',
                    'strength': 'strong' if abs(c) > 0.5 else 'moderate' if abs(c) > 0.2 else 'weak',
                })

        # Key rules from Decision Tree
        dt = self.models.get('Decision Tree')
        if dt and hasattr(dt,'tree_') and names:
            try:
                tree = dt.tree_
                n_nodes = tree.node_count
                feature = tree.feature
                threshold = tree.threshold
                # Get top 3 split features
                splits = {}
                for node in range(n_nodes):
                    if feature[node] >= 0 and feature[node] < len(names):
                        fn = names[feature[node]]
                        splits[fn] = splits.get(fn, 0) + 1
                top_splits = sorted(splits.items(), key=lambda x:-x[1])[:3]
                result['key_rules'] = [{'feature':f,'split_count':c} for f,c in top_splits]
            except Exception:
                pass

        return result
