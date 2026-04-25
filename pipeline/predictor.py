"""
Predictor - Generate submission from test data
Uses optimal threshold per model if available.
"""
import pandas as pd
import numpy as np

DEFAULT_THRESHOLD = 0.45

def generate_submission(test_df, preprocessor, models, results, model_name='best'):
    ids = test_df['enrollee_id'].values.copy()
    X_test = preprocessor.transform(test_df)

    # Select model
    model_keys = [k for k in results if not k.startswith('_')]
    if model_name == 'best':
        chosen = max(model_keys, key=lambda k: results[k].get('roc_auc', 0))
    elif model_name in models:
        chosen = model_name
    else:
        chosen = model_keys[0]

    clf = models[chosen]
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = clf.predict(X_test).astype(float)

    submission = pd.DataFrame({'enrollee_id': ids,
                                'target': np.round(y_proba, 4)})

    opt_t = results.get(chosen, {}).get('opt_threshold', DEFAULT_THRESHOLD)
    if opt_t is None:
        opt_t = DEFAULT_THRESHOLD
    meta  = {
        'model_used':        chosen,
        'total_predictions': int(len(submission)),
        'predicted_seekers': int((y_proba >= opt_t).sum()),
        'predicted_stayers': int((y_proba < opt_t).sum()),
        'seek_rate':         round(float((y_proba >= opt_t).mean() * 100), 2),
        'opt_threshold':     opt_t,
        'avg_confidence':    round(float(np.abs(y_proba - 0.5).mean()), 4),
        'val_metrics': {k: v for k, v in results.get(chosen, {}).items()
                        if k in ('accuracy','precision','recall','f1',
                                 'roc_auc','f1_tuned','cv_f1_mean','overfit_flag')},
    }
    return submission, meta
