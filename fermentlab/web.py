import io
import json
import traceback
import os
import tempfile
import threading

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import batches as BT
from . import config as C
from . import cupping as CP
from . import data as D
from . import forecast as FC
from . import phases as PH
from . import settings as ST
from . import simulate as SIM
from . import windows as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get('FERMENT_MODELS', os.path.join(ROOT, 'models'))
SAMPLE = os.path.join(ROOT, 'data', 'sample_fermentor.csv')
CH = list(C.CONTINUOUS) + ['co2_sat', 'motor']
MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', 50))
SERVE_ONLY = os.environ.get('SERVE_ONLY', '').lower() in ('1', 'true', 'yes')
router = APIRouter(prefix='/api/lab')
cache = {}
state = {'run': None, 'step': C.BASE_STEP_MIN, 'cup_kind': None}
jobs = {k: {'state': 'idle', 'pct': 0, 'msg': '', 'result': None} for k in ('forecast', 'cupping')}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_result(name, res):
    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, f'result_{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, default=_jsonable)


def save_state():
    os.makedirs(MODELS, exist_ok=True)
    run = state['run']
    if run is not None:
        run.to_csv(os.path.join(MODELS, 'last_run.csv'))
    with open(os.path.join(MODELS, 'state.json'), 'w') as f:
        json.dump({'step': state['step'], 'cup_kind': state['cup_kind']}, f)


def restore():
    for name in ('forecast', 'cupping'):
        path = os.path.join(MODELS, f'result_{name}.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                jobs[name].update(state='done', pct=100, msg='Loaded saved results', result=json.load(f))
    sp = os.path.join(MODELS, 'state.json')
    if os.path.exists(sp):
        with open(sp) as f:
            meta = json.load(f)
        state.update(step=meta.get('step', C.BASE_STEP_MIN), cup_kind=meta.get('cup_kind'))
    rp = os.path.join(MODELS, 'last_run.csv')
    if os.path.exists(rp):
        state['run'] = pd.read_csv(rp, index_col='time', parse_dates=True)


def records(df):
    return df.round(3).astype(object).where(df.notna(), None).to_dict('records')


def sources(items):
    out, tmp = [], []
    for name, raw in items:
        if name.lower().endswith('.zip'):
            f = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            f.write(raw)
            f.close()
            out.append(f.name)
            tmp.append(f.name)
        else:
            out.append(io.BytesIO(raw))
    return out, tmp


def load_clean(items):
    srcs, tmp = sources(items)
    try:
        return D.clean(D.read_logs(srcs))
    finally:
        for p in tmp:
            os.remove(p)


def future_grid(run, fc):
    f = fc.set_index('horizon_min')
    grid = np.arange(0, 361, 10)
    xs = [0] + list(f.index)
    data = {c: np.interp(grid, xs, [run[c].iloc[-1]] + list(f[c])) for c in W.TARGETS}
    return pd.DataFrame(data, index=run.index[-1] + pd.to_timedelta(grid, unit='min'))


def virtual_batch(run, fut, i):
    add = fut.iloc[1:i + 1].copy()
    add['co2_sat'] = (add['co2'] >= C.CO2_SATURATION).astype(float)
    add['motor'] = 0.0
    agg = {c: 'mean' for c in CH}
    agg['motor'] = 'max'
    return D.add_derived(pd.concat([run[CH], add[CH]]).resample('30min').agg(agg).dropna())


def advise(run, fut, target_ph, cm):
    grid = [int((t - fut.index[0]).total_seconds() / 60) for t in fut.index]
    if cm is not None:
        s = cm.predict({str(i): virtual_batch(run, fut, i) for i in range(1, len(fut))})['Average'].values
        k = int(np.argmax(s))
        return {'mode': 'cupping', 'best_min': grid[k + 1], 'best_score': float(s[k]), 'scores': [float(x) for x in s]}
    out = {'mode': 'rule', 'eta_min': None}
    if target_ph:
        sign = 1.0 if run['ph'].iloc[-1] > target_ph else -1.0
        hit = np.flatnonzero(sign * (fut['ph'].values - target_ph) <= 0)
        if len(hit):
            out['eta_min'] = grid[int(hit[0])]
    return out


def cupping_model():
    path = os.path.join(MODELS, 'cupping_model.joblib')
    if 'cm' not in cache and os.path.exists(path):
        cache['cm'] = CP.CuppingModel.load(path)
    return cache.get('cm')


def forecast_model():
    path = os.path.join(MODELS, 'forecaster.joblib')
    if 'fm' not in cache and os.path.exists(path):
        cache['fm'] = FC.ForecastModel.load(path)
    return cache.get('fm')


def forecast_payload(run, fm, which=None):
    fc = fm.predict_tail({'run': run}, which)
    fut = future_grid(run, fc)
    hist = run.iloc[-288:]
    return {
        'duration_h': float((run.index[-1] - run.index[0]).total_seconds() / 3600),
        'now': {c: float(run[c].iloc[-1]) for c in CH},
        'history': {'t': [t.isoformat() for t in hist.index], **{c: hist[c].round(4).tolist() for c in W.TARGETS}},
        'future': {'minutes': list(range(0, 361, 10)), **{c: fut[c].round(4).tolist() for c in W.TARGETS}},
        'forecast': records(fc.drop(columns=['run']).assign(forecast_time=fc['forecast_time'].astype(str))),
    }


def run_forecast(items, source, quality, user):
    j = jobs['forecast']
    try:
        j.update(state='running', pct=2, msg='Reading and cleaning data', result=None)
        cfg, used = ST.resolve('forecast', quality, user)
        df = load_clean([('s.csv', open(SAMPLE, 'rb').read())] if source == 'sample' else items)
        runs = D.prepare_runs(df, cfg.step_min)
        total = cfg.n_folds * (len(cfg.tab_models) + len(cfg.archs)) + 1
        done = [0]

        def prog(m):
            done[0] += 1
            j.update(pct=min(96, 5 + int(90 * done[0] / total)), msg=m)

        exp = FC.run_experiment(runs, cfg, progress=prog)
        os.makedirs(MODELS, exist_ok=True)
        exp['model'].save(os.path.join(MODELS, 'forecaster.joblib'))
        exp['summary'].to_csv(os.path.join(MODELS, 'forecast_summary.csv'))
        run = runs[list(runs)[-1]]
        state.update(run=run, step=cfg.step_min)
        cache.pop('fm', None)
        res = forecast_payload(run, exp['model'])
        try:
            lab, seg, k = PH.detect_phases({'run': run})
            ph = {'n': k, 'segments': records(seg[['segment', 'phase', 'start', 'end', 'hours', 'ph_mean', 'co2_mean', 'o2_mean', 'temp_process_mean']].astype({'start': str, 'end': str})),
                  'labels': [int(v) for v in lab['run'].values[::3]]}
        except Exception:
            ph = {'n': 0, 'segments': [], 'labels': []}
        res.update(phases=ph, settings=used, preset=quality, config={'sensor_step_min': cfg.step_min},
                   summary=records(exp['summary'].reset_index()), detail=records(exp['detail']),
                   backtest=exp['backtest'], folds_table=records(exp['folds']),
                   info={'windows': exp['windows'], 'tested': exp['tested'], 'folds': exp['splits'],
                         'runs': len(runs), 'glitches': df.attrs['glitches'], 'best_tab': exp['best_tab'],
                         'best_seq': exp['best_seq']})
        save_result('forecast', res)
        save_state()
        j.update(state='done', pct=100, msg='Completed', result=res)
    except Exception as e:
        j.update(state='error', msg=f'{type(e).__name__}: {e}')


def batch_summary(runs):
    rows = []
    for k, g in runs.items():
        rows.append({'session': k, 'hours': round((g.index[-1] - g.index[0]).total_seconds() / 3600.0, 1),
                     'ph_start': g['ph'].iloc[0], 'ph_end': g['ph'].iloc[-1], 'ph_min': g['ph'].min(),
                     'co2_max': g['co2'].max(), 'o2_min': g['o2'].min(), 'temp_mean': g['temp_process'].mean(),
                     'temp_max': g['temp_process'].max(), 'pressure_max': g['pressure'].max()})
    return pd.DataFrame(rows)


def batch_traces(runs, points=60, limit=40):
    out = {}
    for k in list(runs)[:limit]:
        g = runs[k]
        idx = np.unique(np.linspace(0, len(g) - 1, min(points, len(g))).astype(int))
        out[k] = {'hours': ((g.index[idx] - g.index[0]).total_seconds() / 3600.0).round(2).tolist(),
                  **{c: g[c].iloc[idx].round(3).tolist() for c in ('ph', 'co2', 'o2', 'temp_process', 'pressure')}}
    return out


def run_cupping(items, cup_raw, source, quality, user):
    j = jobs['cupping']
    try:
        j.update(state='running', pct=2, msg='Reading and cleaning data', result=None)
        cfg, used = ST.resolve('cupping', quality, user)
        if source == 'sample':
            raw, win = SIM.simulate(24, 0)
            df = D.clean(D.read_logs(io.BytesIO(raw.to_csv(index=False).encode())))
            runs = D.prepare_runs(df, cfg.step_min, windows=win)
            scores = D.load_scores(io.StringIO(win.to_csv(index=False)))
            info = {'kind': 'simulated', 'n_real_batches': len(runs), 'n_used': len(runs), 'windows_defined': True,
                    'notes': ['24 simulated batches with simulated scores.']}
        else:
            df = load_clean(items)
            runs, scores, info = BT.resolve(df, cup_raw, cfg.step_min)
        windows = BT.windows_table(runs, scores)
        total = 2 + len(cfg.models) + len(cfg.archs)
        done = [0]

        def prog(m):
            done[0] += 1
            j.update(pct=min(94, 5 + int(88 * done[0] / total)), msg=m)

        exp = CP.evaluate(runs, scores, cfg, progress=prog)
        j.update(pct=96, msg='Fitting final model')
        model = CP.fit_final(exp, cfg)
        os.makedirs(MODELS, exist_ok=True)
        model.save(os.path.join(MODELS, 'cupping_model.joblib'))
        exp['results'].to_csv(os.path.join(MODELS, 'cupping_results.csv'), index=False)
        cache['cm'] = model
        state['cup_kind'] = info['kind']
        best = exp['results'].iloc[0]['Model']
        imp = CP.importance_table(model.tab, exp['F'], exp['y']).head(15).reset_index().rename(columns={'index': 'feature'})
        per_batch = pd.DataFrame({'session': exp['ids'], 'actual': exp['y'], **{k: v for k, v in exp['oof'].items()}})
        per_batch['best_model_error'] = (per_batch[best] - per_batch['actual']).abs()
        used_runs = {k: runs[k] for k in exp['ids']}
        final = model.predict(used_runs)
        final['actual'] = exp['y']
        final = final.reset_index()
        try:
            clusters = CP.cluster_batches(exp['F'], exp['y'], exp['ids'])
        except Exception:
            clusters = None
        res = {'results': records(exp['results']), 'cv': exp['cv'], 'n': len(exp['ids']), 'best': best,
               'oof': {'ids': exp['ids'], 'y': exp['y'].tolist(), 'pred': exp['oof'][best].tolist()},
               'importance': records(imp), 'simulated': info['kind'] == 'simulated',
               'info': info, 'batches': records(windows), 'per_batch': records(per_batch),
               'summary': records(batch_summary(used_runs).merge(windows[['session', 'score']], on='session', how='left')),
               'traces': batch_traces(used_runs), 'final': records(final), 'clusters': clusters,
               'glitches': df.attrs.get('glitches', {}), 'duration_h': float(sum((g.index[-1] - g.index[0]).total_seconds() for g in used_runs.values()) / 3600.0),
               'settings': used, 'preset': quality, 'models_fitted': {'tabular': model.tab_name, 'sequence': model.seq_name}}
        save_result('cupping', res)
        save_state()
        j.update(state='done', pct=100, msg='Completed', result=res)
    except Exception as e:
        j.update(state='error', msg=f'{type(e).__name__}: {e}')


async def collect(files):
    return [(f.filename, await f.read()) for f in files]


def parse_settings(raw):
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise ST.SettingsError('The advanced settings could not be read.')
    return data if isinstance(data, dict) else {}


def too_big(items):
    return sum(len(r) for _, r in items) > MAX_UPLOAD_MB * 1024 * 1024


@router.get('/saved')
def saved():
    return {'serve_only': SERVE_ONLY, 'forecast': jobs['forecast']['result'] is not None,
            'cupping': jobs['cupping']['result'] is not None}


@router.get('/settings')
def settings():
    return ST.public()


@router.post('/forecast/train')
async def forecast_train(files: list[UploadFile] = File(default=[]), source: str = Form('upload'),
                         quality: str = Form('fast'), settings: str = Form('')):
    if SERVE_ONLY:
        return JSONResponse({'error': 'Training is disabled on this server. Models were trained offline.'}, status_code=403)
    if jobs['forecast']['state'] == 'running':
        return JSONResponse({'error': 'A training job is already running'}, status_code=409)
    if source != 'sample' and not files:
        return JSONResponse({'error': 'Select a data file first'}, status_code=422)
    items = await collect(files)
    if too_big(items):
        return JSONResponse({'error': f'The files are larger than {MAX_UPLOAD_MB} MB'}, status_code=413)
    try:
        user = parse_settings(settings)
        ST.resolve('forecast', quality, user)
    except ST.SettingsError as e:
        return JSONResponse({'error': str(e)}, status_code=422)
    threading.Thread(target=run_forecast, args=(items, source, quality, user), daemon=True).start()
    return {'ok': True}


@router.post('/cupping/train')
async def cupping_train(files: list[UploadFile] = File(default=[]), cupping: UploadFile = File(None),
                        source: str = Form('upload'), quality: str = Form('fast'), settings: str = Form('')):
    if SERVE_ONLY:
        return JSONResponse({'error': 'Training is disabled on this server. Models were trained offline.'}, status_code=403)
    if jobs['cupping']['state'] == 'running':
        return JSONResponse({'error': 'A training job is already running'}, status_code=409)
    if source != 'sample' and not files:
        return JSONResponse({'error': 'Select the sensor log files first (same format as Mode 1)'}, status_code=422)
    items = await collect(files)
    if too_big(items):
        return JSONResponse({'error': f'The files are larger than {MAX_UPLOAD_MB} MB'}, status_code=413)
    cup = await cupping.read() if cupping is not None else b''
    err = BT.cupping_problem(cup) if source != 'sample' else None
    if err:
        return JSONResponse({'error': err}, status_code=422)
    try:
        user = parse_settings(settings)
        ST.resolve('cupping', quality, user)
    except ST.SettingsError as e:
        return JSONResponse({'error': str(e)}, status_code=422)
    threading.Thread(target=run_cupping, args=(items, cup, source, quality, user), daemon=True).start()
    return {'ok': True}


def reg_rows(out, thr):
    """Error statistics of every prediction column against the real cupping scores."""
    ok = out['actual'].notna()
    if not ok.any():
        return []
    y = out.loc[ok, 'actual'].values.astype(float)
    rows = []
    for c in [c for c in out.columns if c.endswith('(Tabular)') or c.endswith('(Sequence)') or c == 'Average']:
        p = out.loc[ok, c].values.astype(float)
        e = p - y
        sst = float(((y - y.mean()) ** 2).sum())
        rows.append({'model': c, 'batches': int(ok.sum()), 'MAE': float(np.abs(e).mean()),
                     'RMSE': float(np.sqrt((e ** 2).mean())), 'bias': float(e.mean()),
                     'max_abs_error': float(np.abs(e).max()),
                     'R2': float(1 - (e ** 2).sum() / sst) if sst > 1e-12 and len(y) > 1 else None,
                     'accuracy': float(((p >= thr) == (y >= thr)).mean())})
    return rows


@router.post('/cupping/predict')
async def cupping_predict(files: list[UploadFile] = File(...), cupping: UploadFile = File(None)):
    cm = cupping_model()
    if cm is None:
        return JSONResponse({'error': 'Train Mode 2 first.'}, status_code=422)
    items = await collect(files)
    if too_big(items):
        return JSONResponse({'error': f'The files are larger than {MAX_UPLOAD_MB} MB'}, status_code=413)
    cup_raw = await cupping.read() if cupping is not None else b''
    err = BT.cupping_problem(cup_raw)
    if err:
        return JSONResponse({'error': err}, status_code=422)
    try:
        df = load_clean(items)
        cup = BT.read_cupping(cup_raw)
        notes = []
        runs = None
        if cup is not None and {'start', 'end'} <= set(cup.columns):
            try:
                runs = D.prepare_runs(df, cm.cfg.step_min, windows=cup[['session', 'start', 'end']])
            except Exception:
                notes.append('The start and end times in the cupping file did not match the logs, so batches were split by time gaps.')
        if runs is None:
            runs = BT.renumber(D.prepare_runs(df, cm.cfg.step_min))
        out = cm.predict(runs)
        thr = cm.threshold
        metrics = []
        if cup is not None:
            s = cup.dropna(subset=['score']).drop_duplicates('session', keep='last')
            s = pd.Series(s['score'].astype(float).values, index=s['session'].values)
            out['actual'] = s.reindex(out.index.astype(str)).values
            out['error'] = out['Average'] - out['actual']
            out['abs_error'] = out['error'].abs()
            metrics = reg_rows(out, thr)
            if not out['actual'].notna().any():
                notes.append('None of the sessions in the cupping file matched the detected batches.')
        out = out.reset_index()
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=422)
    return {'predictions': records(out), 'kind': state['cup_kind'], 'batches': records(batch_summary(runs)),
            'metrics': metrics, 'threshold': thr, 'notes': notes}


@router.post('/forecast/predict')
async def forecast_predict(files: list[UploadFile] = File(...), model: str = Form('')):
    fm = forecast_model()
    if fm is None:
        return JSONResponse({'error': 'Train Mode 1 first.'}, status_code=422)
    items = await collect(files)
    if too_big(items):
        return JSONResponse({'error': f'The files are larger than {MAX_UPLOAD_MB} MB'}, status_code=413)
    which = model if model in fm.predictors else fm.default
    try:
        runs = D.prepare_runs(load_clean(items), fm.cfg.step_min)
        L = fm.cfg.lookback_steps
        usable = {k: g for k, g in runs.items() if len(g) >= L}
        if not usable:
            raise ValueError(f'The data is too short. Each run needs at least {fm.cfg.lookback_min} minutes of continuous logs.')
        out_runs = []
        for k, g in list(usable.items())[-5:]:
            out_runs.append({'name': str(k), 'start': str(g.index[0]), 'end': str(g.index[-1]), **forecast_payload(g, fm, which)})
        try:
            bt, bt_note = fm.backtest(usable, which), None
        except ValueError:
            bt, bt_note = None, ('The runs are too short to compare against what actually happened. At least '
                                 f'{fm.cfg.lookback_min + max(fm.cfg.horizons_min)} minutes per run are needed.')
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=422)
    return {'model': which, 'models': list(fm.predictors), 'step_min': fm.cfg.step_min, 'runs': out_runs,
            'backtest': bt, 'backtest_note': bt_note, 'skipped': len(runs) - len(usable)}


@router.get('/cupping/example.csv')
def cupping_example():
    return PlainTextResponse('session,score\n1,82.5\n2,79.0\n3,84.0\n4,77.5\n5,81.0\n6,83.5\n',
                             media_type='text/csv',
                             headers={'Content-Disposition': 'attachment; filename="cupping_example.csv"'})


@router.get('/status/{name}')
def status(name: str):
    return jobs.get(name, {'state': 'idle'})


@router.post('/merge')
def merge(target_ph: float = Form(None)):
    fpath = os.path.join(MODELS, 'forecaster.joblib')
    if state['run'] is None or not os.path.exists(fpath):
        return JSONResponse({'error': 'Train Mode 1 first to obtain sensor forecasts.'}, status_code=422)
    try:
        cm = cupping_model()
        if cm is None:
            return JSONResponse({'error': 'Train Mode 2 first to obtain cup-quality estimates.'}, status_code=422)
        if 'fm' not in cache:
            cache['fm'] = FC.ForecastModel.load(fpath)
        fc = cache['fm'].predict_tail({'run': state['run']})
        fut = future_grid(state['run'], fc)
        return {'cup_kind': state['cup_kind'], 'advice': advise(state['run'], fut, target_ph, cm),
                'future': {'minutes': list(range(0, 361, 10)), **{c: fut[c].round(4).tolist() for c in W.TARGETS}}}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'error': f'{type(e).__name__}: {e}'}, status_code=500)


@router.post('/merge-logs')
async def merge_logs(files: list[UploadFile] = File(...)):
    df = load_clean(await collect(files))
    back = {v: k for k, v in C.RAW_COLUMNS.items()}
    out = df[[c for c in back if c in df.columns]].rename(columns=back)
    out['uptime_s'] = 0
    return PlainTextResponse(out.to_csv(index=False), media_type='text/csv')


restore()


def sim_cache():
    if 'sim' not in cache:
        cache['sim'] = SIM.simulate(24, 0)
    return cache['sim']


@router.get('/sample/logs.csv')
def sample_logs(mode: int = 2):
    if mode == 1:
        return PlainTextResponse(open(SAMPLE).read(), media_type='text/csv')
    return PlainTextResponse(sim_cache()[0].to_csv(index=False), media_type='text/csv')


@router.get('/sample/batches.csv')
def sample_batches():
    return PlainTextResponse(sim_cache()[1].to_csv(index=False), media_type='text/csv')
