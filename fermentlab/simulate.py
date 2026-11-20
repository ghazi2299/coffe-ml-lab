import numpy as np
import pandas as pd


def simulate(n_runs=24, seed=0, step_min=5, start='2026-11-01 00:00:00'):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    frames, windows = [], []
    for k in range(1, n_runs + 1):
        hours = rng.uniform(40, 90)
        n = int(hours * 60 / step_min)
        t = np.arange(n) * step_min / 60.0
        drop = rng.uniform(0.1, 0.8)
        tau = rng.uniform(8, 30)
        ph = 6.55 - drop * (1 - np.exp(-t / tau)) + rng.normal(0, 0.012, n)
        growth = rng.uniform(300, 1500)
        co2 = np.minimum(39998.0, 500 + growth * t + rng.normal(0, 60, n))
        o2d = rng.uniform(2, 9)
        o2 = 20.8 - o2d * (1 - np.exp(-t / rng.uniform(5, 25))) + rng.normal(0, 0.12, n)
        base = rng.uniform(24.5, 28.0)
        temp = base + 0.7 * np.sin(2 * np.pi * (t + rng.uniform(0, 24)) / 24) + 0.004 * t + rng.normal(0, 0.05, n)
        scd = temp + 2.0 + rng.normal(0, 0.15, n)
        sht = temp + 0.1 + rng.normal(0, 0.12, n)
        rh_scd = 86 + rng.normal(0, 0.6, n)
        rh_sht = 88 + rng.normal(0, 0.8, n)
        pressure = np.clip(0.03 + rng.normal(0, 0.012, n), 0, None)
        motor = np.zeros(n)
        per = int(6 * 60 / step_min)
        motor[per::per] = 1
        score = (80 + 14 * (drop - 0.45) - 1.2 * (base - 26.0) - 0.5 * (o2d - 5.5)
                 + 0.002 * (growth - 900) / 10 - 0.03 * (hours - 65) + rng.normal(0, 1.2))
        score = float(np.clip(score, 60, 95))
        times = t0 + pd.to_timedelta(np.arange(n) * step_min, unit='min')
        frames.append(pd.DataFrame({
            'timestamp': times.strftime('%Y-%m-%d %H:%M:%S'), 'uptime_s': (np.arange(n) * step_min * 60),
            'suhu_proses_C': temp.round(2), 'ph': ph.round(3), 'tekanan_bar': pressure.round(3),
            'o2_pct': o2.round(2), 'co2_ppm': co2.round(0), 'scd_suhu_C': scd.round(2),
            'scd_rh_pct': rh_scd.round(1), 'sht_suhu_C': sht.round(2), 'sht_rh_pct': rh_sht.round(1),
            'motor': motor.astype(int)}))
        windows.append({'session': k, 'start': str(times[0]), 'end': str(times[-1]), 'score': round(score, 1)})
        t0 = times[-1] + pd.Timedelta(hours=4)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(windows)
