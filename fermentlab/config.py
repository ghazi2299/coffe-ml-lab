from dataclasses import dataclass, field

RAW_COLUMNS = {
    'timestamp': 'time',
    'suhu_proses_C': 'temp_process',
    'ph': 'ph',
    'tekanan_bar': 'pressure',
    'o2_pct': 'o2',
    'co2_ppm': 'co2',
    'scd_suhu_C': 'temp_scd',
    'scd_rh_pct': 'rh_scd',
    'sht_suhu_C': 'temp_sht',
    'sht_rh_pct': 'rh_sht',
    'motor': 'motor',
}

CONTINUOUS = ['temp_process', 'ph', 'pressure', 'o2', 'co2', 'temp_scd', 'rh_scd', 'temp_sht', 'rh_sht']

VALID_RANGE = {
    'temp_process': (0.0, 80.0),
    'ph': (1.0, 13.0),
    'pressure': (-0.5, 5.0),
    'o2': (0.0, 21.5),
    'co2': (0.0, 40000.0),
    'temp_scd': (-10.0, 80.0),
    'rh_scd': (0.0, 100.0),
    'temp_sht': (-10.0, 80.0),
    'rh_sht': (0.0, 100.0),
}

SPIKE_CHANNELS = {'o2': 6.0, 'ph': 8.0, 'pressure': 8.0}

CO2_SATURATION = 39000.0
GOOD_THRESHOLD = 80.0
GAP_MINUTES = 60.0
MIN_BATCH_HOURS = 2.0
BASE_STEP_MIN = 5

HORIZONS_MIN = (30, 60, 120, 180, 360)
LOOKBACK_MIN = 360


@dataclass
class ForecastConfig:
    step_min: int = BASE_STEP_MIN
    lookback_min: int = LOOKBACK_MIN
    horizons_min: tuple = HORIZONS_MIN
    n_folds: int = 5
    scheme: str = 'rolling'
    min_train_frac: float = 0.4
    tab_models: tuple = ('Ridge', 'RandomForest', 'ExtraTrees', 'HistGB')
    archs: tuple = ('gru', 'lstm', 'tcn')
    rf_trees: int = 300
    rf_depth: int = 0          # 0 = unlimited
    et_trees: int = 300
    gb_iter: int = 250
    gb_depth: int = 4
    gb_lr: float = 0.05
    noise: float = 0.03
    epochs: int = 150
    patience: int = 20
    seeds: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    hidden: int = 64
    dropout: float = 0.15
    val_frac: float = 0.2
    calibrate: bool = True
    tune: bool = True
    random_state: int = 42

    @property
    def lookback_steps(self):
        return self.lookback_min // self.step_min

    @property
    def horizon_steps(self):
        return tuple(h // self.step_min for h in self.horizons_min)


@dataclass
class CuppingConfig:
    threshold: float = GOOD_THRESHOLD
    repeats: int = 5
    n_splits: int = 5
    step_min: int = 30
    seq_steps: int = 48
    models: tuple = ('Ridge', 'PLS', 'SVR', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'GP')
    archs: tuple = ('gru', 'lstm', 'cnn')
    rf_trees: int = 500
    rf_depth: int = 0          # 0 = unlimited
    et_trees: int = 600
    gb_trees: int = 250
    gb_depth: int = 2
    gb_lr: float = 0.04
    epochs: int = 200
    seeds: int = 5
    lr: float = 3e-3
    weight_decay: float = 1e-2
    hidden: int = 32
    dropout: float = 0.25
    batch_size: int = 16
    noise: float = 0.05
    use_embeddings: bool = True
    tune: bool = True
    random_state: int = 42
