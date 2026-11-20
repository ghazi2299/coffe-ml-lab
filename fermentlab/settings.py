"""Training settings for both modes: schema (rendered by the UI), presets, validation.

The UI asks `/api/lab/settings` for the schema and presets, so this file is the single source of truth.
Every value sent back by the UI is checked and clamped here before it reaches a config dataclass.
"""
from . import config as C


def _o(*pairs):
    return [{'v': v, 'l': l} for v, l in pairs]


def _f(key, label, kind, group, hint='', **kw):
    return {'key': key, 'label': label, 'type': kind, 'group': group, 'hint': hint, **kw}


TAB_OPTS_1 = _o(('Ridge', 'Ridge'), ('RandomForest', 'Random Forest'), ('ExtraTrees', 'Extra Trees'), ('HistGB', 'Gradient Boosting'))
SEQ_OPTS_1 = _o(('gru', 'GRU'), ('lstm', 'LSTM'), ('tcn', 'TCN'))
TAB_OPTS_2 = _o(('Ridge', 'Ridge'), ('PLS', 'PLS'), ('SVR', 'SVR'), ('RandomForest', 'Random Forest'),
                ('ExtraTrees', 'Extra Trees'), ('GradientBoosting', 'Gradient Boosting'), ('GP', 'Gaussian Process'))
SEQ_OPTS_2 = _o(('gru', 'GRU'), ('lstm', 'LSTM'), ('cnn', '1D-CNN'))

TABULAR = 'Tabular Machine Learning'
DEEP = 'Deep Learning'
VALID = 'Validation'

SCHEMA = {
    'forecast': [
        _f('tab_models', 'Models', 'multi', TABULAR, options=TAB_OPTS_1),
        _f('rf_trees', 'Random Forest Trees', 'int', TABULAR, min=20, max=1000, step=10),
        _f('rf_depth', 'Random Forest Max Depth', 'int', TABULAR, '0 = unlimited', min=0, max=30, step=1),
        _f('et_trees', 'Extra Trees Estimators', 'int', TABULAR, min=20, max=1000, step=10),
        _f('gb_iter', 'Boosting Iterations', 'int', TABULAR, min=20, max=800, step=10),
        _f('gb_depth', 'Boosting Max Depth', 'int', TABULAR, min=1, max=10, step=1),
        _f('gb_lr', 'Boosting Learning Rate', 'float', TABULAR, min=0.005, max=0.5, step=0.005),
        _f('tune', 'Tune Hyperparameters', 'bool', TABULAR, 'Grid search, slower.'),
        _f('archs', 'Models', 'multi', DEEP, 'Leave empty to skip.', options=SEQ_OPTS_1),
        _f('epochs', 'Epochs', 'int', DEEP, min=5, max=400, step=5),
        _f('patience', 'Early Stopping Patience', 'int', DEEP, min=3, max=100, step=1),
        _f('seeds', 'Ensemble Seeds', 'int', DEEP, min=1, max=8, step=1),
        _f('lr', 'Learning Rate', 'float', DEEP, min=0.0001, max=0.05, step=0.0001),
        _f('weight_decay', 'Weight Decay', 'float', DEEP, min=0, max=0.1, step=0.0005),
        _f('batch_size', 'Batch Size', 'int', DEEP, min=8, max=512, step=8),
        _f('hidden', 'Hidden Units', 'int', DEEP, min=8, max=256, step=8),
        _f('dropout', 'Dropout', 'float', DEEP, min=0, max=0.6, step=0.01),
        _f('noise', 'Noise Augmentation', 'float', DEEP, min=0, max=0.3, step=0.01),
        _f('horizons_min', 'Forecast Horizons', 'multi', VALID, '6 h is always included.', options=_o(*[(m, f'{m} min') for m in (30, 60, 120, 180, 240, 360)])),
        _f('lookback_min', 'Look-Back Window (min)', 'int', VALID, min=60, max=720, step=30),
        _f('scheme', 'Validation Scheme', 'choice', VALID, options=_o(('rolling', 'Rolling Origin'), ('expanding', 'Expanding Blocks'), ('loro', 'Leave One Run Out'))),
        _f('n_folds', 'Validation Folds', 'int', VALID, min=2, max=8, step=1),
        _f('min_train_frac', 'Minimum Training Fraction', 'float', VALID, 'Rolling origin only.', min=0.2, max=0.7, step=0.05),
        _f('val_frac', 'Validation Tail Fraction', 'float', VALID, min=0.1, max=0.4, step=0.05),
        _f('calibrate', 'Shrinkage Calibration', 'bool', VALID),
        _f('random_state', 'Random Seed', 'int', VALID, min=0, max=9999, step=1),
    ],
    'cupping': [
        _f('models', 'Models', 'multi', TABULAR, options=TAB_OPTS_2),
        _f('rf_trees', 'Random Forest Trees', 'int', TABULAR, min=20, max=1500, step=10),
        _f('rf_depth', 'Random Forest Max Depth', 'int', TABULAR, '0 = unlimited', min=0, max=30, step=1),
        _f('et_trees', 'Extra Trees Estimators', 'int', TABULAR, min=20, max=1500, step=10),
        _f('gb_trees', 'Boosting Trees', 'int', TABULAR, min=20, max=800, step=10),
        _f('gb_depth', 'Boosting Max Depth', 'int', TABULAR, min=1, max=8, step=1),
        _f('gb_lr', 'Boosting Learning Rate', 'float', TABULAR, min=0.005, max=0.5, step=0.005),
        _f('tune', 'Tune Hyperparameters', 'bool', TABULAR, 'Grid search, slower.'),
        _f('archs', 'Models', 'multi', DEEP, 'Leave empty to skip.', options=SEQ_OPTS_2),
        _f('epochs', 'Epochs', 'int', DEEP, min=5, max=500, step=5),
        _f('seeds', 'Ensemble Seeds', 'int', DEEP, min=1, max=8, step=1),
        _f('lr', 'Learning Rate', 'float', DEEP, min=0.0001, max=0.05, step=0.0001),
        _f('weight_decay', 'Weight Decay', 'float', DEEP, min=0, max=0.1, step=0.0005),
        _f('batch_size', 'Batch Size', 'int', DEEP, min=4, max=128, step=4),
        _f('hidden', 'Hidden Units', 'int', DEEP, min=8, max=128, step=8),
        _f('dropout', 'Dropout', 'float', DEEP, min=0, max=0.6, step=0.01),
        _f('noise', 'Noise Augmentation', 'float', DEEP, min=0, max=0.3, step=0.01),
        _f('threshold', 'Acceptable Score Threshold', 'float', VALID, 'Specialty coffee starts at 80.', min=60, max=95, step=0.5),
        _f('n_splits', 'Cross-Validation Folds', 'int', VALID, 'Fewer than 10 batches use leave-one-out.', min=2, max=10, step=1),
        _f('repeats', 'Cross-Validation Repeats', 'int', VALID, min=1, max=8, step=1),
        _f('random_state', 'Random Seed', 'int', VALID, min=0, max=9999, step=1),
    ],
}

PRESETS = {
    'forecast': {
        'fast': dict(tab_models=['Ridge', 'RandomForest', 'ExtraTrees'], archs=[], rf_trees=150, rf_depth=0, et_trees=200,
                     gb_iter=250, gb_depth=4, gb_lr=0.05, tune=False, epochs=40, patience=20, seeds=1, lr=0.002,
                     weight_decay=0.001, batch_size=64, hidden=64, dropout=0.15, noise=0.03,
                     horizons_min=[30, 60, 120, 180, 360], lookback_min=360, scheme='rolling', n_folds=3,
                     min_train_frac=0.4, val_frac=0.2, calibrate=True, random_state=42),
        'full': dict(tab_models=['Ridge', 'RandomForest', 'ExtraTrees', 'HistGB'], archs=['gru', 'lstm', 'tcn'],
                     rf_trees=300, rf_depth=0, et_trees=300, gb_iter=250, gb_depth=4, gb_lr=0.05, tune=True,
                     epochs=100, patience=20, seeds=2, lr=0.002, weight_decay=0.001, batch_size=64, hidden=64,
                     dropout=0.15, noise=0.03, horizons_min=[30, 60, 120, 180, 360], lookback_min=360,
                     scheme='rolling', n_folds=3, min_train_frac=0.4, val_frac=0.2, calibrate=True, random_state=42),
    },
    'cupping': {
        'fast': dict(models=['Ridge', 'PLS', 'SVR', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'GP'],
                     rf_trees=300, rf_depth=0, et_trees=300, gb_trees=150, gb_depth=2, gb_lr=0.04, tune=False,
                     archs=['gru', 'lstm'], epochs=60, seeds=2, lr=0.003, weight_decay=0.01, batch_size=16,
                     hidden=32, dropout=0.25, noise=0.05, threshold=80.0, n_splits=5, repeats=2, random_state=42),
        'full': dict(models=['Ridge', 'PLS', 'SVR', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'GP'],
                     rf_trees=500, rf_depth=0, et_trees=600, gb_trees=250, gb_depth=2, gb_lr=0.04, tune=True,
                     archs=['gru', 'lstm', 'cnn'], epochs=150, seeds=4, lr=0.003, weight_decay=0.01, batch_size=16,
                     hidden=32, dropout=0.25, noise=0.05, threshold=80.0, n_splits=5, repeats=4, random_state=42),
    },
}

CONFIG_CLASS = {'forecast': C.ForecastConfig, 'cupping': C.CuppingConfig}
TUPLE_KEYS = {'tab_models', 'archs', 'models', 'horizons_min'}


class SettingsError(ValueError):
    pass


def _coerce(field, value):
    t, key = field['type'], field['key']
    if t == 'bool':
        return value if isinstance(value, bool) else str(value).lower() in ('1', 'true', 'yes', 'on')
    if t == 'multi':
        if not isinstance(value, (list, tuple)):
            raise SettingsError(f'{field["label"]}: expected a list')
        allowed = {o['v'] for o in field['options']}
        picked = [v for v in value if v in allowed]
        return [o['v'] for o in field['options'] if o['v'] in picked]
    if t == 'choice':
        allowed = {o['v'] for o in field['options']}
        if value not in allowed:
            raise SettingsError(f'{field["label"]}: unknown option {value!r}')
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise SettingsError(f'{field["label"]}: not a number')
    v = min(max(v, field['min']), field['max'])
    return int(round(v)) if t == 'int' else v


def resolve(kind, quality, user=None):
    """Merge preset and user overrides, validate, and return (config dataclass, plain dict that was used)."""
    base = dict(PRESETS[kind].get(quality, PRESETS[kind]['fast']))
    fields = {f['key']: f for f in SCHEMA[kind]}
    for k, v in (user or {}).items():
        if k in fields:
            base[k] = v
    used = {k: _coerce(fields[k], v) for k, v in base.items()}
    if kind == 'forecast':
        if not used['tab_models'] and not used['archs']:
            raise SettingsError('Select at least one model.')
        used['horizons_min'] = sorted(set(used['horizons_min']) | {360})   # 6 h is needed for the stop advisory
        used['lookback_min'] = max(60, used['lookback_min'] // C.BASE_STEP_MIN * C.BASE_STEP_MIN)
    else:
        if not used['models'] and not used['archs']:
            raise SettingsError('Select at least one model.')
    cfg = CONFIG_CLASS[kind](**{k: (tuple(v) if k in TUPLE_KEYS else v) for k, v in used.items()})
    return cfg, used


def public():
    return {'schema': SCHEMA, 'presets': PRESETS}
