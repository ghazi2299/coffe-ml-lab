import argparse
import os
import sys
import warnings

import pandas as pd

from . import config as C
from . import cupping as CP
from . import data as D
from . import forecast as FC
from . import phases as PH
from . import simulate as SIM


def _say(msg):
    print(msg, flush=True)


def _load_clean(paths):
    df = D.clean(D.read_logs(paths))
    _say(f'{len(df):,} rows, {df.time.iloc[0]} -> {df.time.iloc[-1]}')
    glitches = df.attrs['glitches']
    _say(f'glitches replaced: {glitches}')
    return df


def cmd_forecast(a):
    cfg = C.ForecastConfig(epochs=a.epochs, seeds=a.seeds, n_folds=a.folds, scheme=a.scheme,
                           tune=not a.fast, archs=tuple(a.archs), tab_models=tuple(a.tab), lookback_min=a.lookback,
                           horizons_min=tuple(a.horizons))
    df = _load_clean(a.logs)
    windows = pd.read_csv(a.windows) if a.windows else None
    runs = D.prepare_runs(df, cfg.step_min, windows=windows)
    _say(f'{len(runs)} run(s): ' + ', '.join(f'{k}={len(v)} pts' for k, v in runs.items()))
    exp = FC.run_experiment(runs, cfg, progress=_say)
    os.makedirs(a.out, exist_ok=True)
    exp['summary'].to_csv(os.path.join(a.out, 'forecast_summary.csv'))
    exp['detail'].to_csv(os.path.join(a.out, 'forecast_detail.csv'), index=False)
    exp['model'].save(os.path.join(a.out, 'forecaster.joblib'))
    exp['model'].predict_tail(runs).to_csv(os.path.join(a.out, 'forecast_next.csv'), index=False)
    pd.set_option('display.width', 200)
    w, t, k, per = exp['windows'], exp['tested'], exp['splits'], exp['period_steps']
    _say(f'\nwindows={w} tested={t} folds={k} motor_period_steps={per}')
    _say(exp['summary'].round(3).to_string())
    _say(f'\nsaved to {a.out}')


def cmd_cupping(a):
    cfg = C.CuppingConfig(repeats=a.repeats, epochs=a.epochs, seeds=a.seeds, tune=not a.fast, archs=tuple(a.archs))
    df = _load_clean(a.logs)
    win = pd.read_csv(a.windows)
    runs = D.prepare_runs(df, cfg.step_min, windows=win)
    scores = D.load_scores(a.windows)
    forecaster, fine = None, None
    if a.forecaster:
        forecaster = FC.ForecastModel.load(a.forecaster)
        fine = D.prepare_runs(df, forecaster.cfg.step_min, windows=win)
    exp = CP.evaluate(runs, scores, cfg, forecaster, fine, progress=_say)
    os.makedirs(a.out, exist_ok=True)
    exp['results'].to_csv(os.path.join(a.out, 'cupping_results.csv'), index=False)
    model = CP.fit_final(exp, cfg)
    model.save(os.path.join(a.out, 'cupping_model.joblib'))
    CP.importance_table(model.tab, exp['F'], exp['y']).to_csv(os.path.join(a.out, 'cupping_importance.csv'))
    model.predict(runs).to_csv(os.path.join(a.out, 'cupping_predictions.csv'))
    pd.set_option('display.width', 200)
    nb, cvd = len(exp['ids']), exp['cv']
    _say(f'\nbatches={nb} validation={cvd}')
    _say(exp['results'][['Approach', 'Model', 'MAE', 'RMSE', 'R2', 'Spearman', 'Acc_good']].round(3).to_string())
    _say(f'\nsaved to {a.out}')


def cmd_phases(a):
    df = _load_clean(a.logs)
    runs = D.prepare_runs(df, C.BASE_STEP_MIN)
    labels, table, k = PH.detect_phases(runs)
    os.makedirs(a.out, exist_ok=True)
    table.to_csv(os.path.join(a.out, 'phases.csv'), index=False)
    pd.concat(labels.values(), keys=labels.keys()).to_csv(os.path.join(a.out, 'phase_labels.csv'))
    _say(f'{k} phases\n' + table.round(2).to_string(index=False))


def cmd_simulate(a):
    raw, win = SIM.simulate(a.runs, a.seed)
    os.makedirs(a.out, exist_ok=True)
    raw.to_csv(os.path.join(a.out, 'sim_logs.csv'), index=False)
    win.to_csv(os.path.join(a.out, 'sim_batches.csv'), index=False)
    _say(f'wrote {a.out}/sim_logs.csv and {a.out}/sim_batches.csv')


def main(argv=None):
    warnings.filterwarnings('ignore')
    p = argparse.ArgumentParser(prog='fermentlab')
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('forecast')
    f.add_argument('--logs', nargs='+', required=True)
    f.add_argument('--windows')
    f.add_argument('--out', default='out_forecast')
    f.add_argument('--scheme', default='rolling', choices=['rolling', 'blocked', 'loro'])
    f.add_argument('--folds', type=int, default=5)
    f.add_argument('--epochs', type=int, default=150)
    f.add_argument('--seeds', type=int, default=3)
    f.add_argument('--archs', nargs='*', default=['gru', 'tcn'])
    f.add_argument('--tab', nargs='*', default=['Ridge', 'ExtraTrees', 'HistGB'])
    f.add_argument('--lookback', type=int, default=C.LOOKBACK_MIN)
    f.add_argument('--horizons', nargs='+', type=int, default=list(C.HORIZONS_MIN))
    f.add_argument('--fast', action='store_true')
    f.set_defaults(fn=cmd_forecast)
    c = sub.add_parser('cupping')
    c.add_argument('--logs', nargs='+', required=True)
    c.add_argument('--windows', required=True)
    c.add_argument('--forecaster')
    c.add_argument('--out', default='out_cupping')
    c.add_argument('--repeats', type=int, default=5)
    c.add_argument('--epochs', type=int, default=200)
    c.add_argument('--seeds', type=int, default=5)
    c.add_argument('--archs', nargs='*', default=['gru', 'cnn'])
    c.add_argument('--fast', action='store_true')
    c.set_defaults(fn=cmd_cupping)
    ph = sub.add_parser('phases')
    ph.add_argument('--logs', nargs='+', required=True)
    ph.add_argument('--out', default='out_phases')
    ph.set_defaults(fn=cmd_phases)
    s = sub.add_parser('simulate')
    s.add_argument('--runs', type=int, default=24)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--out', default='sim')
    s.set_defaults(fn=cmd_simulate)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
