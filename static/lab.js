(() => {
const $ = s => document.querySelector(s);
const NAME = {ph: 'pH', co2: 'CO₂', o2: 'O₂', temp_process: 'Suhu Proses', pressure: 'Tekanan', temp_scd: 'Suhu Chamber (SCD)', rh_scd: 'Kelembapan (SCD)', temp_sht: 'Suhu (SHT)', rh_sht: 'Kelembapan (SHT)'};
const UNIT = {ph: '', co2: 'ppm', o2: '%', temp_process: '°C', pressure: 'bar', temp_scd: '°C', rh_scd: '%', temp_sht: '°C', rh_sht: '%'};
const BASE = ['Persistence', 'RollingMean12'];
let mode = null, src = 'upload', files = [], R = {}, cur = 'ph', timer = null, SET = null;

setTimeout(() => $('#splash').classList.add('out'), 3400);
$('#splash').addEventListener('click', () => $('#splash').classList.add('out'));


let SERVE = false, QUIET = false;
const fmtMin = m => { const h = Math.floor(m / 60), r = m % 60; return h ? `${h} jam${r ? ' ' + r + ' mnt' : ''}` : `${r} mnt`; };
const num = (c, v) => v.toFixed(c === 'co2' ? 0 : c === 'pressure' ? 3 : 2);
const pct = v => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
const ACR = {mae: 'MAE', rmse: 'RMSE', nmae: 'nMAE', r2: 'R²', ph: 'pH', co2: 'CO₂', o2: 'O₂', rh: 'RH', scd: 'SCD', sht: 'SHT', cv: 'CV', acc: 'Akurasi', persist: 'Baseline', oof: 'Out-of-Fold', h: 'jam'};
const HEAD = {horizon_min: 'Horizon (mnt)', Horizon_min: 'Horizon (mnt)', Skill_persist: 'Skill vs Baseline', Skill_roll: 'Skill vs Rata-rata Bergerak', Skill_Versus_Naive: 'Skill vs Baseline Naif', Skill_persist_median: 'Median Skill vs Baseline', R2_delta: 'R² (Perubahan)', Corr: 'Korelasi', Cells_beat_persist: 'Prakiraan Lebih Baik dari Baseline', Acc_good: 'Akurasi', Acc_good_std: 'Std Akurasi', F1_good: 'F1 (Layak)', F1_good_std: 'Std F1', MAE_std: 'Std MAE', RMSE_std: 'Std RMSE', R2_std: 'Std R²', Spearman_std: 'Std Spearman', abs_error: 'Error Mutlak', error_pct: 'Error (%)', Model_gap: 'Selisih Model', skill_vs_persistence: 'Skill vs Baseline', persistence_MAE: 'MAE Baseline', max_abs_error: 'Error Maksimum', min_24h: 'Min (24 jam)', max_24h: 'Maks (24 jam)', mean_24h: 'Rata-rata (24 jam)', train_windows: 'Jendela Latih', test_windows: 'Jendela Uji', predicted_acceptable: 'Prediksi Layak', predicted_below: 'Prediksi Di Bawah Standar'};
const PCT = new Set(['Skill_persist', 'Skill_roll', 'Skill_persist_median', 'Skill_Versus_Naive', 'Cells_beat_persist', 'Acc_good', 'Acc_good_std', 'accuracy', 'skill_vs_persistence']);
const SIGNED = new Set(['Skill_persist', 'Skill_roll', 'Skill_persist_median', 'Skill_Versus_Naive', 'skill_vs_persistence']);
const head = k => HEAD[k] || String(k).replace(/[|]/g, ' ').split(/[_\s]+/).filter(Boolean).map(w => ACR[w.toLowerCase()] || (w[0].toUpperCase() + w.slice(1))).join(' ');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const CSVS = {};
let csvN = 0;
function colFmt(rows, k, pctAll) {
  const vals = rows.map(r => r[k]).filter(v => v !== null && v !== undefined && v !== '');
  if (!vals.length || !vals.every(isNum)) return null;
  if (pctAll || PCT.has(k)) return v => `${SIGNED.has(k) || pctAll ? (v > 0 ? '+' : '') : ''}${(v * 100).toFixed(1)}%`;
  if (vals.every(Number.isInteger)) return v => String(v);
  const mx = Math.max(...vals.map(Math.abs)), d = mx >= 100 ? 2 : mx >= 0.1 ? 3 : 4;
  return v => v.toFixed(d);
}
function tbl(rows, o = {}) {
  if (!rows || !rows.length) return '<p class="mu">Tidak ada data.</p>';
  const keys = Object.keys(rows[0]), fm = {}, best = {};
  keys.forEach(k => {
    fm[k] = colFmt(rows, k, o.pct);
    if (o.best && fm[k]) { const v = rows.map(r => r[k]).filter(isNum); best[k] = o.best === 'min' ? Math.min(...v) : Math.max(...v); }
  });
  const td = (r, k) => {
    const v = r[k];
    if (v === null || v === undefined) return '<td class="n">—</td>';
    if (isNum(v) && fm[k]) {
      const c = ['n', best[k] === v ? 'lo' : '', SIGNED.has(k) || o.pct ? (v >= 0 ? 'pos' : 'neg') : ''].filter(Boolean).join(' ');
      return `<td class="${c}">${fm[k](v)}</td>`;
    }
    return `<td>${esc(NAME[v] ?? v)}</td>`;
  };
  let csv = '';
  if (o.csv) { const id = 'c' + (++csvN); CSVS[id] = {name: o.csv, rows: o.all || rows}; csv = `<button class="lb alt sm" data-csv="${id}">Unduh CSV</button>`; }
  const tall = rows.length > 12;
  return `<div class="sc${tall ? ' tall' : ''}"><table class="ltb"><thead><tr>${keys.map(k => `<th class="${fm[k] ? 'n' : ''}">${esc(head(k))}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${keys.map(k => td(r, k)).join('')}</tr>`).join('')}</tbody></table></div>${tall || csv ? `<div class="tfoot"><span>${rows.length} baris</span>${csv}</div>` : ''}`;
}
document.addEventListener('click', ev => {
  const b = ev.target.closest && ev.target.closest('[data-csv]');
  if (!b || !CSVS[b.dataset.csv]) return;
  const c = CSVS[b.dataset.csv], rows = typeof c.rows === 'function' ? c.rows() : c.rows;
  if (!rows.length) return;
  const keys = Object.keys(rows[0]), q = v => v === null || v === undefined ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v;
  const txt = [keys.join(',')].concat(rows.map(r => keys.map(k => q(r[k])).join(','))).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([txt], {type: 'text/csv'}));
  a.download = c.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

function openMode(m, scroll) {
  mode = m;
  document.querySelectorAll('.mode').forEach(x => x.classList.toggle('on', +x.dataset.m === m));
  $('#data').hidden = false;
  $('#dataHint').textContent = m === 1 ? 'Mode 1 hanya membutuhkan log sensor.' : 'Mode 2 membutuhkan log sensor dan opsional file cupping.';
  $('#tphRow').hidden = m !== 1;
  setSrc(src);
  renderAdv();
  if (R[m]) {
    const q = QUIET;
    QUIET = true;
    m === 1 ? render1() : render2();
    QUIET = q;
  }
  if (scroll) $('#data').scrollIntoView({behavior: 'smooth', block: 'start'});
}
document.querySelectorAll('.mode').forEach(b => b.addEventListener('click', () => {
  if (SERVE) { serveShow(+b.dataset.m); return; }
  openMode(+b.dataset.m, true);
}));

function setSrc(s) {
  src = s;
  document.querySelectorAll('.src button').forEach(b => b.classList.toggle('on', b.dataset.s === s));
  ['api', 'upload', 'sample', 'model'].forEach(k => { $('#s_' + k).hidden = k !== s; });
  $('#cupBox').hidden = !(mode === 2 && s === 'upload');
  $('#goBtn').disabled = s === 'api' || s === 'model';
  if (s === 'model' && mode) loadSaved();
}
async function loadSaved() {
  const name = kindOf(), box = $('#s_model');
  const s = await (await fetch(`/api/lab/status/${name}`)).json();
  if (s.state !== 'done' || !s.result) { box.textContent = 'Belum ada hasil tersimpan untuk mode ini.'; return; }
  box.textContent = 'Menampilkan hasil dari model yang tersimpan di server.';
  R[mode] = s.result;
  mode === 1 ? render1() : render2();
}
async function serveShow(m) {
  mode = m;
  document.querySelectorAll('.mode').forEach(x => x.classList.toggle('on', +x.dataset.m === m));
  ['#out1', '#out2', '#mergeOut'].forEach(i => { $(i).hidden = true; });
  $('#data').hidden = m !== 1;
  $('#tphRow').hidden = m !== 1;
  $('#noSaved').hidden = !!R[m];
  if (R[m]) m === 1 ? render1() : render2();
}
fetch('/api/lab/saved').then(r => r.json()).then(async v => {
  if (!v.forecast && !v.cupping) return;
  for (const [m, k] of [[1, 'forecast'], [2, 'cupping']]) {
    const s = await (await fetch(`/api/lab/status/${k}`)).json();
    if (s.state === 'done' && s.result) R[m] = s.result;
  }
  if (!SET) SET = await (await fetch('/api/lab/settings')).json();
  const first = R[1] ? 1 : 2;
  QUIET = true;
  if (v.serve_only) {
    SERVE = true;
    document.body.classList.add('serve');
    $('#data h3').textContent = 'Target pH';
    $('#dataHint').textContent = 'Opsional. Dipakai untuk rekomendasi waktu henti gabungan.';
    ['goBtn', 'quality'].forEach(i => { const el = $('#' + i); (el.closest('label') || el).hidden = true; });
    await serveShow(first);
  } else {
    openMode(first, false);
  }
  QUIET = false;
}).catch(() => {});
document.querySelectorAll('.src button').forEach(b => b.addEventListener('click', () => setSrc(b.dataset.s)));

const fi = $('#labFile'), dz = $('#drop');
const setFiles = l => { files = [...l]; $('#dropText').textContent = files.length ? files.map(f => `${f.name} (${Math.round(f.size / 1024)} KB)`).join(', ') : 'Log sensor: tarik file .csv atau .zip ke sini, atau klik untuk memilih'; dz.classList.toggle('has', files.length > 0); };
fi.addEventListener('change', () => setFiles(fi.files));
dz.addEventListener('click', () => fi.click());
['dragover', 'dragenter'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove('over'); }));
dz.addEventListener('drop', ev => setFiles(ev.dataTransfer.files));

let cupFile = null;
const cf = $('#cupFile'), cd = $('#cupDrop');
const setCup = f => { cupFile = f || null; $('#cupText').textContent = cupFile ? `${cupFile.name} (${Math.max(1, Math.round(cupFile.size / 1024))} KB)` : 'Skor cupping (opsional): tarik file .csv ke sini, atau klik untuk memilih'; cd.classList.toggle('has', !!cupFile); };
cf.addEventListener('change', () => setCup(cf.files[0]));
cd.addEventListener('click', () => cf.click());
['dragover', 'dragenter'].forEach(e => cd.addEventListener(e, ev => { ev.preventDefault(); cd.classList.add('over'); }));
['dragleave', 'drop'].forEach(e => cd.addEventListener(e, ev => { ev.preventDefault(); cd.classList.remove('over'); }));
cd.addEventListener('drop', ev => setCup(ev.dataTransfer.files[0]));


fetch('/api/lab/settings').then(r => r.json()).then(s => { SET = s; if (mode) renderAdv(); }).catch(() => {});
const kindOf = () => mode === 1 ? 'forecast' : 'cupping';
const fieldOf = k => SET.schema[kindOf()].find(f => f.key === k);

// Label & hint override untuk pengaturan lanjutan (dari schema backend yang berbahasa Inggris)
const FLBL = {
  tab_models: 'Model Tabular', archs: 'Model Deep Learning', models: 'Model',
  rf_trees: 'Jumlah Pohon Random Forest', rf_depth: 'Kedalaman Maks Random Forest',
  et_trees: 'Jumlah Estimator Extra Trees', gb_iter: 'Iterasi Boosting',
  gb_trees: 'Jumlah Pohon Boosting', gb_depth: 'Kedalaman Maks Boosting',
  gb_lr: 'Learning Rate Boosting', tune: 'Optimasi Hyperparameter',
  epochs: 'Epoch', patience: 'Patience Early Stopping', seeds: 'Seed Ensemble',
  lr: 'Learning Rate', weight_decay: 'Weight Decay', batch_size: 'Ukuran Batch',
  hidden: 'Unit Hidden', dropout: 'Dropout', noise: 'Augmentasi Noise',
  horizons_min: 'Horizon Prakiraan', lookback_min: 'Jendela Look-Back (menit)',
  scheme: 'Skema Validasi', n_folds: 'Jumlah Fold Validasi',
  min_train_frac: 'Fraksi Pelatihan Minimum', val_frac: 'Fraksi Validasi Akhir',
  calibrate: 'Kalibrasi Penyusutan', random_state: 'Seed Acak',
  threshold: 'Ambang Skor Layak', n_splits: 'Jumlah Fold Cross-Validation',
  repeats: 'Pengulangan Cross-Validation',
};
const FHINT = {
  rf_depth: '0 = tanpa batas', tune: 'Pencarian grid, lebih lambat.',
  horizons_min: '6 jam selalu disertakan.', min_train_frac: 'Hanya untuk rolling origin.',
  archs: 'Kosongkan untuk melewati.', threshold: 'Kopi specialty dimulai dari skor 80.',
  n_splits: 'Kurang dari 10 batch otomatis memakai leave-one-out.',
};
const FGROUP = {'Tabular Machine Learning': 'Machine Learning Tabular', 'Deep Learning': 'Deep Learning', 'Validation': 'Validasi'};
const FOPT = {'Rolling Origin': 'Rolling Origin', 'Expanding Blocks': 'Blok Ekspansi', 'Leave One Run Out': 'Tinggalkan Satu Run'};
const fLabel = f => FLBL[f.key] || f.label;
const fHint = f => FHINT[f.key] !== undefined ? FHINT[f.key] : f.hint;
const oLabel = o => FOPT[o.l] || o.l;

const hint = f => fHint(f) ? `<small>${fHint(f)}</small>` : '';
function fieldHTML(f) {
  if (f.type === 'multi') return `<div class="af wide"><span class="al">${fLabel(f)}</span><div class="checks">${f.options.map(o => `<label class="ck"><input type="checkbox" data-k="${f.key}" value="${o.v}"> ${oLabel(o)}</label>`).join('')}</div>${hint(f)}</div>`;
  if (f.type === 'bool') return `<label class="af ckrow"><input type="checkbox" data-k="${f.key}"><span>${fLabel(f)}${hint(f)}</span></label>`;
  if (f.type === 'choice') return `<label class="af"><span class="al">${fLabel(f)}</span><select data-k="${f.key}">${f.options.map(o => `<option value="${o.v}">${oLabel(o)}</option>`).join('')}</select>${hint(f)}</label>`;
  return `<label class="af"><span class="al">${fLabel(f)}</span><input type="number" data-k="${f.key}" min="${f.min}" max="${f.max}" step="${f.step}">${hint(f)}</label>`;
}
function renderAdv() {
  if (!SET || !mode) return;
  const fields = SET.schema[kindOf()], groups = [...new Set(fields.map(f => f.group))];
  $('#advBody').innerHTML = groups.map(g => `<fieldset class="ag"><legend>${FGROUP[g] || g}</legend><div class="afs">${fields.filter(f => f.group === g).map(fieldHTML).join('')}</div></fieldset>`).join('');
  if ($('#quality').value === 'custom') $('#quality').value = 'fast';
  $('#quality').querySelector('[value=custom]').hidden = true;
  fillAdv(SET.presets[kindOf()][$('#quality').value]);
}
function fillAdv(vals) {
  document.querySelectorAll('#advBody [data-k]').forEach(el => {
    const f = fieldOf(el.dataset.k), v = vals[el.dataset.k];
    if (f.type === 'multi') el.checked = (v || []).map(String).includes(el.value);
    else if (f.type === 'bool') el.checked = !!v;
    else el.value = v;
  });
}
function collectAdv() {
  if (!SET || !mode) return {};
  const o = {};
  SET.schema[kindOf()].forEach(f => {
    const els = [...document.querySelectorAll(`#advBody [data-k="${f.key}"]`)];
    if (f.type === 'multi') o[f.key] = els.filter(e => e.checked).map(e => typeof f.options[0].v === 'number' ? +e.value : e.value);
    else if (f.type === 'bool') o[f.key] = els[0].checked;
    else if (f.type === 'choice') o[f.key] = els[0].value;
    else o[f.key] = els[0].value === '' ? f.min : +els[0].value;
  });
  return o;
}
$('#advBody').addEventListener('input', () => { const q = $('#quality'), c = q.querySelector('[value=custom]'); c.hidden = false; q.value = 'custom'; $('#advHint').textContent = 'Kustom'; });
$('#quality').addEventListener('change', () => { if (SET && mode && $('#quality').value !== 'custom') { fillAdv(SET.presets[kindOf()][$('#quality').value]); $('#advHint').textContent = ''; } });
$('#advReset').addEventListener('click', () => { const q = $('#quality'); q.value = 'fast'; q.querySelector('[value=custom]').hidden = true; fillAdv(SET.presets[kindOf()].fast); $('#advHint').textContent = ''; });

function settingRows(r) {
  const k = r === R[1] ? 'forecast' : 'cupping';
  const rows = [{parameter: 'Preset', value: {fast: 'Standar', full: 'Akurasi Tinggi', custom: 'Kustom'}[r.preset] || r.preset}];
  SET.schema[k].forEach(f => {
    const v = r.settings[f.key];
    const val = f.type === 'multi' ? ((v || []).map(x => oLabel(f.options.find(o => o.v === x) || {l: x})).join(', ') || 'tidak ada') : f.type === 'bool' ? (v ? 'ya' : 'tidak') : f.type === 'choice' ? oLabel(f.options.find(o => o.v === v) || {l: v}) : v;
    rows.push({parameter: f.type === 'multi' ? (f.group.startsWith('Tabular') ? 'Model tabular' : f.group.startsWith('Deep') ? 'Model sekuens' : fLabel(f)) : fLabel(f), value: val});
  });
  return rows;
}
function mountTabs(root, names, body, after) {
  root.innerHTML = `<div class="chips tabs2">${names.map((t, i) => `<button class="chip${i ? '' : ' on'}" data-k="${t}">${t}</button>`).join('')}</div><div class="tbody"></div>`;
  const show = k => {
    root.querySelector('.tbody').innerHTML = body[k]();
    root.querySelectorAll('.tabs2 .chip').forEach(b => b.classList.toggle('on', b.dataset.k === k));
    if (after) after(k, root);
  };
  root.querySelectorAll('.tabs2 .chip').forEach(b => b.addEventListener('click', () => show(b.dataset.k)));
  show(names[0]);
}

$('#goBtn').addEventListener('click', async () => {
  if (!mode) return;
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  if (mode === 2 && cupFile) fd.append('cupping', cupFile);
  fd.append('source', src);
  fd.append('quality', $('#quality').value);
  fd.append('settings', JSON.stringify(collectAdv()));
  const name = mode === 1 ? 'forecast' : 'cupping';
  const r = await fetch(`/api/lab/${name}/train`, {method: 'POST', body: fd});
  if (!r.ok) { alert((await r.json()).error || 'Tidak dapat memulai pelatihan.'); return; }
  $('#goBtn').disabled = true;
  $('#prog').hidden = false;
  $(`#out${mode}`).hidden = true;
  $('#progTitle').textContent = mode === 1 ? 'Melatih Model Prakiraan Sensor' : 'Melatih Model Skor Cupping';
  $('#prog').scrollIntoView({behavior: 'smooth', block: 'center'});
  const m = mode;
  clearInterval(timer);
  timer = setInterval(async () => {
    const s = await (await fetch(`/api/lab/status/${name}`)).json();
    $('#progBar').style.width = s.pct + '%';
    $('#progMsg').textContent = s.state === 'error' ? `Gagal: ${s.msg}` : `${s.pct}%: ${s.msg}`;
    if (s.state === 'done' || s.state === 'error') {
      clearInterval(timer);
      $('#goBtn').disabled = false;
      if (s.state === 'done') { R[m] = s.result; setTimeout(() => { $('#prog').hidden = true; m === 1 ? render1() : render2(); }, 600); }
    }
  }, 900);
});

function skills(model) {
  const o = {};
  R[1].detail.filter(d => d.Model === model).forEach(d => { (o[d.Channel] = o[d.Channel] || []).push(d.Skill_persist); });
  return Object.entries(o).map(([c, v]) => [c, v.reduce((a, b) => a + b, 0) / v.length]).sort((a, b) => b[1] - a[1]);
}
const PAL = ['#7A4326', '#46633C', '#C8A27C', '#94452A', '#5B6B8C', '#B08A3E'];

function render1() {
  const r = R[1], sum = r.summary, best = sum.find(x => !BASE.includes(x.Model)), sk = skills(best.Model);
  const f = r.future, last = f.minutes.length - 1;
  $('#out1').hidden = false;
  
  // Kotak panduan yang sudah dirapikan
  const guide = `<div class="note" style="background:#FCF9F3; border-left-color:var(--accent); margin-bottom:20px; padding:16px 20px;">
    <div style="font:600 15px var(--serif); margin-bottom:8px; color:var(--ink);">💡 Panduan Singkat:</div>
    <p style="margin:0; line-height:1.6; color:var(--muted);">Di bawah ini adalah ringkasan performa AI. Untuk melihat <strong>angka ramalan (prediksi)</strong> suhu, pH, dan gas ke depan, silakan klik tombol <strong>👁️ Lihat Hasil Prediksi (Grafik)</strong>.</p>
  </div>`;

  $('#sum1').innerHTML = `${guide}
  <p class="mu">Model dilatih dari ${r.duration_h.toFixed(1)} jam data (${r.info.windows} jendela). Prakiraan mencakup ${Object.keys(NAME).length} kanal sensor hingga 6 jam ke depan.</p>
  <div class="lgrid">
    <div class="stat"><span>Algoritma Terpintar</span><b>${best.Model}</b><span>Tingkat error ${best.nMAE.toFixed(2)} (makin kecil makin akurat)</span></div>
    <div class="stat"><span>Tingkat Kecerdasan AI</span><b>${pct(best.Skill_persist)}</b><span>Lebih pintar dari tebakan biasa pada ${(best.Cells_beat_persist * 100).toFixed(0)}% kondisi</span></div>
    <div class="stat"><span>Sensor Paling Stabil</span><b>${NAME[sk[0][0]]}</b><span>Paling mudah diprediksi (skor ${pct(sk[0][1])})</span></div>
    <div class="stat"><span>Sensor Paling Sulit</span><b>${NAME[sk[sk.length - 1][0]]}</b><span>Paling sulit ditebak (skor ${pct(sk[sk.length - 1][1])})</span></div>
    <div class="stat"><span>Prediksi pH (6 Jam)</span><b>${num('ph', f.ph[last])}</b><span>Nilai saat ini ${num('ph', r.now.ph)}</span></div>
    <div class="stat"><span>Prediksi Suhu (6 Jam)</span><b>${num('temp_process', f.temp_process[last])} °C</b><span>Nilai saat ini ${num('temp_process', r.now.temp_process)} °C</span></div>
  </div>
  ${best.Skill_persist < 0.05 ? '<p class="mu" style="margin-top:12px;color:var(--poor)">Catatan: AI hanya sedikit lebih pintar dari tebakan biasa. Semakin banyak data fermentasi, semakin pintar AI ini.</p>' : ''}`;

  $('#viz1').innerHTML = `<p class="mu">Garis coklat: data pengamatan masa lalu. Garis putus-putus hijau: <b>ramalan/prediksi AI</b>. Garis merah: waktu henti yang kamu pilih.</p><div class="chips" id="chips"></div><div class="cwrap" id="chart"></div><p style="margin:14px 0 4px"><b>Jika Fermentasi Dihentikan: <span id="stopLbl">Sekarang</span></b></p><input class="lr" type="range" id="stopRange" min="0" max="360" step="10" value="0"><div class="sv" id="stopVals"></div><h3 style="margin:22px 0 4px">Aktual vs Prediksi</h3><div id="vbt"></div>`;
  
  $('#stopRange').addEventListener('input', () => { chart(); stopInfo(); });
  drawChips(); chart(); stopInfo();
  
  r.backtest ? btView($('#vbt'), r.backtest, {model: r.info.best_tab && r.info.best_seq ? 'Ensemble' : (r.info.best_tab || r.info.best_seq), note: 'Validasi: setiap titik diprediksi oleh AI yang belum pernah melihat bagian data itu.'}) : $('#vbt').innerHTML = '<p class="mu">Latih ulang model untuk menampilkan tabel ini.</p>';
  
  detail1(); predPanel(1);
  if (!QUIET) $('#out1').scrollIntoView({behavior: 'smooth'});
}

const approachOf = (m, r) => BASE.includes(m) ? 'Baseline' : m === 'Ensemble' ? 'Ensemble' : ((r.settings || {}).tab_models || []).includes(m) ? 'Tabular' : 'Deep Learning';
const avg = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);

function detail1() {
  const r = R[1], tabs = ['Data', 'Parameter', 'Perbandingan', 'Pelatihan', 'Aktual vs Prediksi', 'Fase', 'Prakiraan'], chans = Object.keys(NAME);
  const step = (r.config && r.config.sensor_step_min || 5) / 60;
  const mini = (c, x, y, xl) => `<div class="mini"><b>${NAME[c]}${unitTxt(c)}</b><small>Terakhir ${num(c, y[y.length - 1])} · Min ${num(c, Math.min(...y))} · Maks ${num(c, Math.max(...y))}</small>${plot([{x, y, c: '#7A4326', w: 2.2}], 300, 170, xl, '', false, {nx: 5})}</div>`;
  const grid = '<div class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">';
  const dm = r.info.best_tab && r.info.best_seq ? `Ensemble dari ${r.info.best_tab} dan ${r.info.best_seq}` : (r.info.best_tab || r.info.best_seq);
  const matrix = (key, agg, o) => { const ms = r.summary.map(x => x.Model); return ms.map(m => { const row = {Model: m}; chans.forEach(c => { const v = r.detail.filter(d => d.Model === m && d.Channel === c).map(d => d[key]).filter(isNum); row[NAME[c]] = v.length ? agg(v) : null; }); return row; }); };
  const body = {
    Data: () => `<p class="mu">Data 24 jam terakhir setelah pembersihan. Angka di bawah tiap grafik menunjukkan nilai terakhir, terendah, dan tertinggi.</p>${grid}${chans.map(c => mini(c, r.history[c].map((_, i) => (i - (r.history[c].length - 1)) * step), r.history[c], 'Jam sebelum sekarang')).join('')}</div>
<h3>Ringkasan Kanal (24 Jam Terakhir)</h3><p class="mu">Nilai terkini, terendah, tertinggi, dan rata-rata tiap sensor.</p>${tbl(chans.map(c => { const a = r.history[c]; return {channel: c, unit: UNIT[c] || '—', now: r.now[c], min_24h: Math.min(...a), max_24h: Math.max(...a), mean_24h: avg(a)}; }))}
<h3>Ringkasan Data</h3><p class="mu">Jumlah run, durasi total, dan jendela yang tersedia untuk pelatihan.</p>${tbl([{runs: r.info.runs, duration_h: +r.duration_h.toFixed(1), windows: r.info.windows, tested_windows: r.info.tested}])}
<h3>Pembersihan Data</h3><p class="mu">Jumlah sampel yang diganti oleh filter glitch, per kanal.</p>${tbl(Object.entries(r.info.glitches).map(([k, v]) => ({channel: k, samples_replaced: v})))}`,
    Parameter: () => `<h3>Konfigurasi</h3><p class="mu">Nilai pengaturan yang dipakai saat pelatihan.</p>${tbl(settingRows(r).concat([{parameter: 'Interval Sampling Sensor', value: r.config.sensor_step_min + ' menit'}]))}`,
    Perbandingan: () => `<h3>Peringkat Model</h3><p class="mu">Error ternormalisasi (nMAE), lebih rendah lebih baik. Baseline adalah prakiraan naif (nilai dianggap tidak berubah) dan rata-rata bergerak.</p>${bars(r.summary, 'Model', 'nMAE')}<div style="height:14px"></div>${tbl(r.summary.map(x => { const {Model, ...rest} = x; return {Model, Approach: approachOf(Model, r), ...rest}; }), {csv: 'perbandingan_model.csv'})}
<h3>nMAE per Model dan Kanal</h3><p class="mu">Dirata-rata di semua horizon. Nilai terendah di tiap kolom disorot.</p>${tbl(matrix('nMAE', avg), {best: 'min'})}
<h3>Skill vs Baseline per Model dan Kanal</h3><p class="mu">Nilai positif berarti lebih baik dari asumsi "nilai tidak berubah".</p>${tbl(matrix('Skill_persist', avg), {pct: true})}
<h3>nMAE per Horizon</h3><p class="mu">Semakin jauh horizon, biasanya error semakin besar.</p>${(() => { const hs = [...new Set(r.detail.map(d => d.Horizon_min))].sort((a, b) => a - b), ms = r.summary.map(x => x.Model); const rows = ms.map(m => { const row = {Model: m}; hs.forEach(h => { row[fmtMin(h)] = avg(r.detail.filter(d => d.Model === m && d.Horizon_min === h).map(d => d.nMAE)); }); return row; }); return `${legend(ms.map((m, i) => [m, MPAL[i % 10]]))}${plot(ms.map((m, i) => ({x: hs, y: hs.map(h => avg(r.detail.filter(d => d.Model === m && d.Horizon_min === h).map(d => d.nMAE))), c: MPAL[i % 10], pts: true, dash: BASE.includes(m), tips: hs.map(h => `${m} pada ${fmtMin(h)}`)})), 900, 300, 'Horizon prakiraan (menit)', 'nMAE', false, {nx: 7})}<div style="height:10px"></div>${tbl(rows, {best: 'min'})}`; })()}
<h3>Error per Kanal dan Horizon</h3><p class="mu">Pilih model untuk melihat detail error tiap kanal dan horizon.</p><div class="row" style="margin:0 0 8px"><label>Model<select id="dm">${[...new Set(r.detail.map(d => d.Model))].map(m => `<option>${m}</option>`).join('')}</select></label></div><div id="dt"></div>`,
    Pelatihan: () => `<h3>Protokol Validasi</h3><p class="mu">Validasi rolling-origin dengan jeda sebesar horizon terpanjang, supaya tidak ada informasi masa depan yang bocor ke pelatihan.</p>${tbl([{windows: r.info.windows, tested: r.info.tested, folds: r.info.folds, lookback_min: r.settings.lookback_min, horizons_min: (r.settings.horizons_min || []).join(', '), scheme: r.settings.scheme}])}
<h3>Fold</h3><p class="mu">Pembagian data latih dan uji di tiap fold.</p>${r.folds_table ? tbl(r.folds_table) : '<p class="mu">Detail fold akan tersimpan pada pelatihan berikutnya.</p>'}
<h3>Model Final</h3><p class="mu">Model dilatih ulang dengan seluruh jendela setelah validasi selesai.</p>${tbl([{role: 'Tabular terbaik', model: r.info.best_tab || '—'}, {role: 'Deep learning terbaik', model: r.info.best_seq || '—'}, {role: 'Model default', model: dm || '—'}])}`,
    'Aktual vs Prediksi': () => r.backtest ? `<h3>Aktual vs Prediksi (Validasi)</h3><p class="mu">Setiap titik diprediksi oleh model yang belum pernah melihat bagian data itu. ${r.backtest.time.length} jendela dengan jarak merata ditampilkan.</p><div id="bt1"></div>` : '<p class="mu">Hasil tersimpan ini dibuat sebelum tabel ini ada. Latih ulang untuk menampilkannya, atau pakai Prediksi Data Baru untuk membandingkan file dengan model yang sudah dilatih.</p>',
    Fase: () => { if (!r.phases.n) return '<p class="mu">Deteksi fase tidak tersedia untuk run ini.</p>'; const lb = r.phases.labels, tot = r.phases.segments.reduce((a, s) => a + s.hours, 0), tk = ticks(0, tot, 7), n = lb.length; return `<p class="mu">Ditemukan ${r.phases.n} fase berdasarkan pengelompokan fitur sensor bergulir.</p>${legend([...new Set(lb)].sort().map(l => [`Fase ${l}`, PAL[l % 6]]))}<svg class="ch" viewBox="0 0 900 62">${lb.map((l, i) => `<rect x="${(14 + i * 872 / n).toFixed(1)}" y="0" width="${(872 / n + 0.6).toFixed(1)}" height="34" fill="${PAL[l % 6]}"/>`).join('')}${tk.t.filter(v => v <= tot).map(v => `<line class="vl" x1="${(14 + v / tot * 872).toFixed(1)}" x2="${(14 + v / tot * 872).toFixed(1)}" y1="34" y2="40"/><text x="${(14 + v / tot * 872).toFixed(1)}" y="54" text-anchor="middle">${v} jam</text>`).join('')}</svg>${tbl(r.phases.segments, {csv: 'fase.csv'})}`; },
    Prakiraan: () => `<h3>Nilai Prakiraan</h3><p class="mu">Model: ${esc(dm)}. Prakiraan dihitung dari ujung run terakhir.</p>${tbl(r.forecast.map(x => { const {horizon_min, forecast_time, ...rest} = x; return {horizon_min, forecast_time, ...rest}; }), {csv: 'nilai_prakiraan.csv'})}
<h3>Kurva Prakiraan</h3><p class="mu">6 jam ke depan, diinterpolasi antar horizon.</p>${grid}${chans.map(c => mini(c, r.future.minutes.map(m => m / 60), r.future[c], 'Jam dari sekarang')).join('')}</div>`,
  };
  mountTabs($('#tech1'), tabs, body, (k, root) => {
    const dsel = root.querySelector('#dm');
    if (dsel) { const dt = () => { root.querySelector('#dt').innerHTML = tbl(r.detail.filter(d => d.Model === dsel.value).map(d => ({Channel: d.Channel, Unit: UNIT[d.Channel] || '—', Horizon_min: d.Horizon_min, MAE: d.MAE, RMSE: d.RMSE, nMAE: d.nMAE, Skill_Versus_Naive: d.Skill_persist})), {csv: 'error_per_kanal_horizon.csv'}); }; dsel.addEventListener('change', dt); dt(); }
    if (k === 'Aktual vs Prediksi' && r.backtest) btView(root.querySelector('#bt1'), r.backtest, {model: r.info.best_tab && r.info.best_seq ? 'Ensemble' : (r.info.best_tab || r.info.best_seq)});
  });
}


const bars = (rows, lab, val, fmt = v => v.toFixed(2)) => { const mx = Math.max(...rows.map(x => x[val]), 1e-9); return rows.map(x => `<div class="brow"><span>${esc(x[lab])}</span><div class="bar"><div style="width:${Math.max(2, x[val] / mx * 100)}%"></div></div><b>${fmt(x[val])}</b></div>`).join(''); };
const CPAL = ['#7A4326', '#46633C', '#5B6B8C', '#B08A3E', '#94452A'];
const cls2 = (v, thr) => v >= thr ? 'Layak' : 'Di bawah standar';

function detail2() {
  const r = R[2], thr = r.settings.threshold, mae = r.results.map(x => ({Model: x.Model, MAE: x.MAE}));
  const pmodels = Object.keys(r.per_batch[0]).filter(k => !['session', 'actual', 'best_model_error'].includes(k));
  const body = {
    Data: () => `<p class="mu">Ringkasan batch yang dipakai untuk pelatihan.</p>${tbl([{batches: r.n, total_hours: +r.duration_h.toFixed(1), score_min: Math.min(...r.oof.y), score_max: Math.max(...r.oof.y), score_mean: +(avg(r.oof.y)).toFixed(1)}])}
<h3>Ringkasan Batch</h3><p class="mu">Durasi, pH awal/akhir, dan kondisi sensor tiap batch.</p>${tbl(r.summary, {csv: 'ringkasan_batch.csv'})}
<h3>Grafik Sensor</h3><p class="mu">Pilih batch untuk melihat jejak sensor tiap batch.</p><div class="row" style="margin:0 0 8px"><label>Batch<select id="tb">${Object.keys(r.traces).map(k => `<option>${k}</option>`).join('')}</select></label></div><div id="tg" class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))"></div>
<h3>Pembersihan Data</h3><p class="mu">Jumlah sampel yang diganti oleh filter glitch, per kanal.</p>${tbl(Object.entries(r.glitches).map(([k, v]) => ({channel: k, samples_replaced: v})))}`,
    Parameter: () => `<h3>Konfigurasi</h3><p class="mu">Nilai pengaturan yang dipakai saat pelatihan.</p>${tbl(settingRows(r))}`,
    Perbandingan: () => `<h3>Peringkat Model</h3><p class="mu">Rata-rata error mutlak dalam poin skor, lebih rendah lebih baik.</p>${bars(mae, 'Model', 'MAE')}<h3>Akurasi Klasifikasi</h3><p class="mu">Persentase batch yang benar diklasifikasikan sebagai layak atau di bawah standar (ambang ${thr}).</p>${bars(r.results.map(x => ({Model: x.Model, A: x.Acc_good})), 'Model', 'A', v => (v * 100).toFixed(0) + '%')}<h3>Semua Metrik</h3><p class="mu">Hasil cross-validation. "Std" adalah rentang antar pengulangan.</p>${tbl(r.results, {csv: 'perbandingan_model.csv'})}`,
    Pelatihan: () => `<h3>Protokol Validasi</h3><p class="mu">${esc(r.cv)} cross-validation: setiap batch dinilai oleh model yang belum pernah melihatnya.</p>${tbl([{batches: r.n, validation: r.cv, final_tabular: r.models_fitted.tabular, final_sequence: r.models_fitted.sequence, threshold: thr}])}<h3>Model Final</h3><p class="mu">Model dilatih ulang dengan seluruh ${r.n} batch setelah validasi. Prediksi keduanya dirata-rata.</p>${tbl([{role: 'Tabular', model: r.models_fitted.tabular || '—'}, {role: 'Sekuens (deep learning)', model: r.models_fitted.sequence || '—'}])}`,
    'Aktual vs Prediksi': () => `<h3>Aktual vs Prediksi (Validasi)</h3><p class="mu">Prediksi out-of-fold: setiap batch diprediksi oleh model yang belum pernah melihatnya. Error = prediksi − aktual, dalam poin skor.</p><div class="row" style="margin:0 0 10px"><label>Model<select id="am">${pmodels.map(m => `<option${m === r.best ? ' selected' : ''}>${esc(m)}</option>`).join('')}</select></label></div><div id="ao"></div><h3>Semua Model Berdampingan</h3><p class="mu">Prediksi skor tiap model untuk setiap batch.</p>${tbl(r.per_batch.map(x => { const o = {session: x.session, actual: x.actual}; pmodels.forEach(m => { o[m] = x[m]; }); return o; }), {csv: 'prediksi_semua_model.csv'})}`,
    'Fitur Penting': () => !r.importance.length ? '<p class="mu">Tidak ada model tabular yang dilatih.</p>' : `<h3>Fitur Paling Berpengaruh</h3><p class="mu">Permutation importance: kenaikan error ketika sebuah fitur diacak. Semakin besar nilainya, semakin penting fitur tersebut.</p>${bars(r.importance.map(x => ({f: x.feature.replace('|', ' '), v: Math.max(0, x.permutation_importance)})), 'f', 'v', v => v.toFixed(3))}<div style="height:14px"></div>${tbl(r.importance, {csv: 'kepentingan_fitur.csv'})}`,
    'Kelompok Batch': () => !r.clusters ? '<p class="mu">Batch belum cukup untuk membentuk kelompok.</p>' : `<h3>Kelompok Batch</h3><p class="mu">Ditemukan ${r.clusters.k} kelompok dengan k-means (silhouette ${r.clusters.silhouette.toFixed(2)}). Sumbu adalah dua komponen utama pertama.</p><div class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"><div>${legend(r.clusters.summary.map(s => [`Kelompok ${s.cluster}`, CPAL[(s.cluster - 1) % 5]]))}${plot([{x: r.clusters.x, y: r.clusters.y, c: '#7A4326', dots: true, cols: r.clusters.labels.map(l => CPAL[(l - 1) % 5]), labels: r.clusters.ids, tips: r.clusters.ids.map((id, i) => `Batch ${id}: kelompok ${r.clusters.labels[i]}, skor ${r.oof.y[i]}`)}], 480, 320, 'Komponen 1', 'Komponen 2')}</div><div>${tbl(r.clusters.summary)}</div></div><h3>Penempatan Batch</h3><p class="mu">Batch mana masuk ke kelompok yang mana.</p>${tbl(r.clusters.ids.map((id, i) => ({session: id, group: r.clusters.labels[i], score: r.oof.y[i]})), {csv: 'kelompok_batch.csv'})}`,
    Prediksi: () => `<h3>Prediksi Model Final</h3><p class="mu">Model final dilatih ulang dengan seluruh ${r.n} batch. Prediksi ini in-sample sehingga terlihat lebih baik dari hasil cross-validation. Untuk menilai batch baru, gunakan Prediksi Data Baru.</p>${tbl(r.final.map(x => { const {actual, ...rest} = x; const e = x.Average - actual; return {...rest, Prediction: x.Prediction === 'Good' ? 'Layak' : 'Di bawah standar', actual, error: e, abs_error: Math.abs(e)}; }), {csv: 'prediksi_final.csv'})}`,
  };
  mountTabs($('#tech2'), Object.keys(body), body, (k, root) => {
    if (k === 'Data') {
      const draw = () => { const t = r.traces[root.querySelector('#tb').value]; root.querySelector('#tg').innerHTML = [['ph', 'pH'], ['co2', 'CO₂'], ['o2', 'O₂'], ['temp_process', 'Suhu Proses'], ['pressure', 'Tekanan']].map(([c, n]) => `<div class="mini"><b>${n}${unitTxt(c)}</b><small>Terakhir ${num(c, t[c][t[c].length - 1])} · Min ${num(c, Math.min(...t[c]))} · Maks ${num(c, Math.max(...t[c]))}</small>${plot([{x: t.hours, y: t[c], c: '#7A4326', w: 2.2}], 300, 170, 'Jam', '', false, {nx: 5})}</div>`).join(''); };
      root.querySelector('#tb').addEventListener('change', draw); draw();
    }
    if (k === 'Aktual vs Prediksi') {
      const draw = () => {
        const m = root.querySelector('#am').value, a = r.per_batch.map(x => x.actual), p = r.per_batch.map(x => x[m]), e = p.map((v, i) => v - a[i]), s = errStats(a, p);
        const ids = r.per_batch.map(x => x.session), tp = a.filter((v, i) => v >= thr && p[i] >= thr).length, fn = a.filter((v, i) => v >= thr && p[i] < thr).length, fp = a.filter((v, i) => v < thr && p[i] >= thr).length, tn = a.filter((v, i) => v < thr && p[i] < thr).length;
        const tips = ids.map((id, i) => `Batch ${id}: aktual ${a[i]}, prediksi ${p[i].toFixed(2)}, error ${e[i] >= 0 ? '+' : ''}${e[i].toFixed(2)}`);
        root.querySelector('#ao').innerHTML = `${tbl([{batches: s.n, MAE: s.mae, RMSE: s.rmse, bias: s.bias, max_abs_error: s.maxe, R2: s.r2, accuracy: (tp + tn) / s.n}])}
<div class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin-top:14px"><div>${legend([['Batch', '#7A4326'], ['Prediksi sempurna', '#8C8C8C']])}${plot([{x: a, y: p, c: '#7A4326', dots: true, labels: ids, tips}], 480, 300, 'Skor aktual', 'Skor prediksi', true)}</div><div>${legend([['Kelebihan prediksi', '#94452A'], ['Kekurangan prediksi', '#5B6B8C']])}${errBars(ids, e, 480, 300)}</div></div>
<h3>Klasifikasi terhadap Ambang ${thr}</h3><p class="mu">Perbandingan hasil klasifikasi layak / di bawah standar antara aktual dan prediksi.</p>${tbl([{actual: 'Layak', predicted_acceptable: tp, predicted_below: fn}, {actual: 'Di bawah standar', predicted_acceptable: fp, predicted_below: tn}])}
<h3>Per Batch</h3><p class="mu">Detail prediksi, error, dan klasifikasi tiap batch.</p>${tbl(ids.map((id, i) => ({session: id, actual: a[i], predicted: p[i], error: e[i], abs_error: Math.abs(e[i]), error_pct: e[i] / a[i] * 100, actual_rating: cls2(a[i], thr), predicted_rating: cls2(p[i], thr), correct: (a[i] >= thr) === (p[i] >= thr) ? 'Ya' : 'Tidak'})), {csv: 'aktual_vs_prediksi.csv'})}`;
      };
      root.querySelector('#am').addEventListener('change', draw); draw();
    }
  });
}

function fileRow(m) {
  return `<div class="row"><label>Log Sensor<input type="file" class="pf" accept=".csv,.zip" multiple></label>${m === 2 ? '<label>Skor Cupping (Opsional)<input type="file" class="pc" accept=".csv"></label>' : `<label>Model<select class="pm">${modelOpts().map(x => `<option value="${esc(x)}">${esc(x)}</option>`).join('')}</select></label>`}<button class="lb pgo">Prediksi</button></div>`;
}
const modelOpts = () => { const i = R[1].info; return [...new Set([i.best_tab && i.best_seq ? 'Ensemble' : null, i.best_tab, i.best_seq].filter(Boolean))]; };

function predPanel(m) {
  const el = $('#pred' + m);
  if (!el || !R[m]) return;
  el.innerHTML = m === 1
    ? `<h3>Prediksi Data Baru</h3><p class="mu">Unggah log sensor baru. Model yang sudah dilatih akan memprakirakan 6 jam ke depan mulai dari ujung tiap run.</p>${fileRow(1)}<div class="pres"></div>`
    : `<h3>Prediksi Data Baru</h3><p class="mu">Unggah log sensor batch baru untuk dinilai dengan model yang sudah dilatih. Tambahkan file cupping berisi skor asli untuk melihat perbandingan aktual vs prediksi dan error.</p>${fileRow(2)}<div class="pres"></div>`;
  const out = el.querySelector('.pres'), go = el.querySelector('.pgo');
  go.addEventListener('click', async () => {
    const f = el.querySelector('.pf').files;
    if (!f.length) { out.innerHTML = '<p class="err">Pilih file log sensor terlebih dahulu.</p>'; return; }
    const fd = new FormData();
    [...f].forEach(x => fd.append('files', x));
    if (m === 1) fd.append('model', el.querySelector('.pm').value);
    else { const c = el.querySelector('.pc').files[0]; if (c) fd.append('cupping', c); }
    out.innerHTML = '<p class="mu">Sedang memproses prediksi…</p>';
    go.disabled = true;
    let res, j;
    try { res = await fetch(m === 1 ? '/api/lab/forecast/predict' : '/api/lab/cupping/predict', {method: 'POST', body: fd}); j = await res.json(); }
    catch (e) { res = {ok: false}; j = {error: 'Tidak dapat menghubungi server.'}; }
    go.disabled = false;
    if (!res.ok) { out.innerHTML = `<p class="err">${esc(j.error || 'Prediksi gagal.')}</p>`; return; }
    m === 1 ? showPred1(out, j) : showPred2(out, j);
  });
}

function showPred1(out, j) {
  const runs = j.runs, step = j.step_min / 60;
  out.innerHTML = `<div class="note"><b>Model yang dipakai: ${esc(j.model)}</b>Prakiraan dimulai dari ujung tiap run yang diunggah.${j.skipped ? ` ${j.skipped} run terlalu pendek dan dilewati.` : ''} Jika file ini bagian dari data pelatihan, hasil perbandingan akan terlihat lebih baik daripada pada data yang benar-benar baru.</div>
<div class="row" style="margin:0 0 8px"><label>Run<select class="pr">${runs.map((x, i) => `<option value="${i}">Run ${esc(x.name)}: ${esc(x.start.slice(0, 16))} hingga ${esc(x.end.slice(0, 16))} (${x.duration_h.toFixed(1)} jam)</option>`).join('')}</select></label></div><div class="chips pch"></div><div class="pcw"></div><div class="ptb"></div><h3 style="margin-top:22px">Aktual vs Prediksi</h3><div class="pbt"></div>`;
  let c = 'ph';
  const draw = () => {
    const x = runs[+out.querySelector('.pr').value], h = x.history[c], n = h.length;
    out.querySelector('.pch').innerHTML = Object.keys(NAME).map(k => `<button class="chip${k === c ? ' on' : ''}" data-c="${k}">${NAME[k]}</button>`).join('');
    out.querySelectorAll('.pch .chip').forEach(b => b.addEventListener('click', () => { c = b.dataset.c; draw(); }));
    out.querySelector('.pcw').innerHTML = `${legend([['Pengamatan', '#7A4326'], ['Prakiraan', '#46633C']])}${plot([{x: h.map((_, i) => (i - (n - 1)) * step), y: h, c: '#7A4326'}, {x: x.future.minutes.map(m => m / 60), y: x.future[c], c: '#46633C', dash: true, w: 2.6}], 900, 300, 'Jam dari ujung run', NAME[c] + unitTxt(c), false, {vlines: [{x: 0, label: 'Ujung run'}], nx: 9})}`;
    out.querySelector('.ptb').innerHTML = `<h3>Nilai Prakiraan</h3><p class="mu">Nilai tiap horizon untuk semua kanal.</p>${tbl(x.forecast.map(f => { const {horizon_min, forecast_time, ...rest} = f; return {horizon_min, forecast_time, ...rest}; }), {csv: 'prakiraan_data_baru.csv'})}`;
  };
  out.querySelector('.pr').addEventListener('change', draw);
  draw();
  const bt = out.querySelector('.pbt');
  if (j.backtest) btView(bt, j.backtest, {model: j.model, note: `${j.backtest.windows} jendela dalam file punya masa depan yang sudah diketahui, dan sudah dibandingkan; ${j.backtest.time.length} jendela dengan jarak merata ditampilkan.`});
  else bt.innerHTML = `<p class="mu">${esc(j.backtest_note || 'Tidak ada perbandingan yang bisa ditampilkan.')}</p>`;
}

function showPred2(out, j) {
  const P = j.predictions, thr = j.threshold, hasA = P.some(x => isNum(x.actual)), avgM = (j.metrics || []).find(x => x.model === 'Average');
  const acc = P.filter(x => x.Average >= thr).length;
  const rows = P.map(x => ({...x, Prediction: x.Prediction === 'Good' ? 'Layak' : 'Di bawah standar'}));
  const ids = P.filter(x => isNum(x.actual)).map(x => x.session), a = P.filter(x => isNum(x.actual)).map(x => x.actual), p = P.filter(x => isNum(x.actual)).map(x => x.Average), e = p.map((v, i) => v - a[i]);
  out.innerHTML = `${(j.notes || []).map(n => `<div class="note">${esc(n)}</div>`).join('')}<div class="lgrid" style="margin:12px 0"><div class="stat"><span>Batch Dinilai</span><b>${P.length}</b><span>${acc} layak, ${P.length - acc} di bawah standar</span></div><div class="stat"><span>Rata-rata Skor Prediksi</span><b>${avg(P.map(x => x.Average)).toFixed(1)}</b><span>Ambang batas ${thr}</span></div>${avgM ? `<div class="stat"><span>Rata-rata Error Mutlak</span><b>${avgM.MAE.toFixed(2)} poin</b><span>${avgM.batches} batch dengan skor asli</span></div><div class="stat"><span>Akurasi Klasifikasi</span><b>${(avgM.accuracy * 100).toFixed(0)}%</b><span>RMSE ${avgM.RMSE.toFixed(2)}, bias ${avgM.bias >= 0 ? '+' : ''}${avgM.bias.toFixed(2)}</span></div>` : ''}</div>
<h3>Skor yang Diprediksi</h3><p class="mu">${hasA ? 'Error = prediksi (rata-rata kedua model) − aktual, dalam poin skor.' : 'Tambahkan file cupping berisi skor asli untuk melihat perbandingan aktual vs prediksi dan error.'}</p>${tbl(rows, {csv: 'prediksi_batch_baru.csv'})}
${hasA ? `<h3>Aktual vs Prediksi</h3><div class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"><div>${legend([['Batch', '#7A4326'], ['Prediksi sempurna', '#8C8C8C']])}${plot([{x: a, y: p, c: '#7A4326', dots: true, labels: ids, tips: ids.map((id, i) => `Batch ${id}: aktual ${a[i]}, prediksi ${p[i].toFixed(2)}`)}], 480, 300, 'Skor aktual', 'Skor prediksi', true)}</div><div>${legend([['Kelebihan prediksi', '#94452A'], ['Kekurangan prediksi', '#5B6B8C']])}${errBars(ids, e, 480, 300)}</div></div><h3>Error per Model</h3><p class="mu">Detail error masing-masing model terhadap skor asli.</p>${tbl(j.metrics)}` : ''}
<h3>Ringkasan Batch</h3><p class="mu">Informasi tiap batch yang baru diproses.</p>${tbl(j.batches)}`;
}

function drawChips() {
  $('#chips').innerHTML = Object.keys(NAME).map(c => `<button class='chip${c === cur ? ' on' : ''}' data-c='${c}'>${NAME[c]}</button>`).join('');
  document.querySelectorAll('#chips .chip').forEach(el => el.addEventListener('click', () => { cur = el.dataset.c; drawChips(); chart(); stopInfo(); }));
}
const stopIdx = () => Math.round(parseInt($('#stopRange').value, 10) / 10);

function chart() {
  const h = R[1].history, f = R[1].future, n = h.t.length, y = h[cur], fy = f[cur];
  const all = y.concat(fy), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo || 1) * 0.12;
  const W = 720, Hh = 280, l = 54, r = 14, t = 12, b = 30, tot = n + 72;
  const X = i => l + (W - l - r) * i / tot, Y = v => t + (Hh - t - b) * (1 - (v - (lo - pad)) / (hi - lo + 2 * pad));
  const pts = y.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const fp = fy.map((v, i) => `${X(n - 1 + i * 2).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  let g = '';
  for (let k = 0; k <= 4; k++) { const v = lo - pad + (hi - lo + 2 * pad) * k / 4; g += `<line x1='${l}' x2='${W - r}' y1='${Y(v)}' y2='${Y(v)}' stroke='var(--line)'/><text x='${l - 6}' y='${Y(v) + 4}' text-anchor='end'>${v.toFixed(Math.abs(v) > 100 ? 0 : 2)}</text>`; }
  const sx = X(n - 1 + stopIdx() * 2);
  $('#chart').innerHTML = `<svg viewBox='0 0 ${W} ${Hh}' width='100%'>${g}<line x1='${X(n - 1)}' x2='${X(n - 1)}' y1='${t}' y2='${Hh - b}' stroke='var(--muted)' stroke-dasharray='3 3'/><text x='${X(n - 1)}' y='${Hh - 10}' text-anchor='middle'>Sekarang</text><text x='${l}' y='${Hh - 10}'>24 jam lalu</text><text x='${W - r}' y='${Hh - 10}' text-anchor='end'>+6 jam</text><polyline points='${pts}' fill='none' stroke='var(--accent)' stroke-width='2.4'/><polyline points='${fp}' fill='none' stroke='var(--good)' stroke-width='2.6' stroke-dasharray='6 5'/><line x1='${sx}' x2='${sx}' y1='${t}' y2='${Hh - b}' stroke='var(--poor)' stroke-width='2'/><circle id='dot' r='4.5' fill='var(--ink)' visibility='hidden'/></svg><div class='tip' id='tip' hidden></div>`;
  const svg = $('#chart svg'), dot = $('#dot'), tip = $('#tip');
  svg.addEventListener('pointermove', ev => {
    const bb = svg.getBoundingClientRect(), px = (ev.clientX - bb.left) / bb.width * W;
    const i = Math.max(0, Math.min(tot - 1, Math.round((px - l) / (W - l - r) * tot))), fut = i >= n;
    const v = fut ? fy[Math.min(fy.length - 1, Math.round((i - n + 1) / 2))] : y[i];
    dot.setAttribute('cx', X(i)); dot.setAttribute('cy', Y(v)); dot.setAttribute('visibility', 'visible');
    tip.hidden = false; tip.style.left = `${X(i) / W * bb.width}px`; tip.style.top = `${Y(v) / Hh * bb.height}px`;
    tip.textContent = `${fut ? 'Dalam ' + fmtMin((i - n + 1) * 5) : fmtMin((n - 1 - i) * 5) + ' lalu'}: ${num(cur, v)} ${UNIT[cur]}${fut ? ' (prakiraan)' : ''}`;
  });
  svg.addEventListener('pointerleave', () => { dot.setAttribute('visibility', 'hidden'); tip.hidden = true; });
}
function stopInfo() {
  const i = stopIdx(), f = R[1].future;
  $('#stopLbl').textContent = i === 0 ? 'Sekarang' : `dalam ${fmtMin(i * 10)}`;
  $('#stopVals').innerHTML = ['ph', 'co2', 'o2', 'temp_process'].map(c => `<div>${NAME[c]}<b>${num(c, f[c][i])}</b>${UNIT[c]}</div>`).join('');
}

function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
  const raw = (hi - lo) / (n - 1), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, t = [];
  for (let v = Math.floor(lo / step + 1e-9) * step; v <= Math.ceil(hi / step - 1e-9) * step + step / 2; v += step) t.push(+v.toFixed(10));
  return {t, step};
}
const tf = step => { const d = Math.max(0, Math.min(4, Math.ceil(-Math.log10(step) - 1e-9))); return v => v.toFixed(d); };
const MPAL = ['#7A4326', '#46633C', '#5B6B8C', '#B08A3E', '#94452A', '#4F8A8B', '#8A5A83', '#6B7F3B', '#8C8C8C', '#C8A27C'];
const legend = items => `<div class="leg2">${items.map(([n, c]) => `<span><i style="background:${c}"></i>${esc(n)}</span>`).join('')}</div>`;

function plot(series, W, H, xl, yl, ref, o = {}) {
  const xs = series.flatMap(s => s.x), ys = series.flatMap(s => s.y).filter(isNum);
  if (!xs.length || !ys.length) return '<p class="mu">Tidak ada data.</p>';
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (ref) { x0 = y0 = Math.min(x0, y0); x1 = y1 = Math.max(x1, y1); }
  const py = (y1 - y0 || 1) * 0.06, px = ref ? (x1 - x0 || 1) * 0.06 : 0;
  const xt = ticks(x0 - px, x1 + px, o.nx || 6), yt = ticks(y0 - py, y1 + py, 5);
  const X0 = xt.t[0], X1 = xt.t[xt.t.length - 1], Y0 = yt.t[0], Y1 = yt.t[yt.t.length - 1];
  const L = 58, R = 14, T = 20, B = 38;
  const X = v => L + (W - L - R) * (v - X0) / (X1 - X0 || 1), Y = v => T + (H - T - B) * (1 - (v - Y0) / (Y1 - Y0 || 1));
  const xf = tf(xt.step), yf = tf(yt.step), r1 = v => v.toFixed(1);
  let g = '';
  yt.t.forEach(v => { g += `<line class="gl" x1="${L}" x2="${W - R}" y1="${r1(Y(v))}" y2="${r1(Y(v))}"/><text x="${L - 7}" y="${r1(Y(v) + 4)}" text-anchor="end">${yf(v)}</text>`; });
  xt.t.forEach(v => { g += `<line class="gl" y1="${T}" y2="${H - B}" x1="${r1(X(v))}" x2="${r1(X(v))}"/><text x="${r1(X(v))}" y="${H - B + 16}" text-anchor="middle">${xf(v)}</text>`; });
  if (ref) g += `<line x1="${X(X0)}" y1="${Y(Y0)}" x2="${X(X1)}" y2="${Y(Y1)}" stroke="var(--muted)" stroke-dasharray="5 4"/>`;
  (o.vlines || []).forEach(v => { g += `<line class="vl" x1="${X(v.x)}" x2="${X(v.x)}" y1="${T}" y2="${H - B}"/><text x="${X(v.x)}" y="${T - 6}" text-anchor="middle">${esc(v.label)}</text>`; });
  let body = '';
  series.forEach(s => {
    const tip = i => s.tips && s.tips[i] ? `<title>${esc(s.tips[i])}</title>` : '';
    if (s.dots) body += s.x.map((v, i) => isNum(s.y[i]) ? `<circle cx="${r1(X(v))}" cy="${r1(Y(s.y[i]))}" r="5" fill="${(s.cols && s.cols[i]) || s.c}" fill-opacity=".8" stroke="var(--surface)" stroke-width="1">${tip(i)}</circle>` : '').join('');
    else {
      const pts = s.x.map((v, i) => isNum(s.y[i]) ? `${r1(X(v))},${r1(Y(s.y[i]))}` : null).filter(Boolean).join(' ');
      body += `<polyline fill="none" stroke="${s.c}" stroke-width="${s.w || 2.4}" stroke-linejoin="round"${s.dash ? ' stroke-dasharray="6 5"' : ''} points="${pts}"/>`;
      if (s.pts) body += s.x.map((v, i) => isNum(s.y[i]) ? `<circle cx="${r1(X(v))}" cy="${r1(Y(s.y[i]))}" r="3" fill="${s.c}">${tip(i)}</circle>` : '').join('');
    }
    if (s.labels) body += s.x.map((v, i) => isNum(s.y[i]) ? `<text class="pt" x="${r1(X(v) + 7)}" y="${r1(Y(s.y[i]) - 6)}">${esc(s.labels[i])}</text>` : '').join('');
  });
  return `<svg class="ch" viewBox="0 0 ${W} ${H}" role="img"><text x="${L - 50}" y="12" text-anchor="start">${esc(yl)}</text>${g}${body}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(xl)}</text></svg>`;
}

function errBars(ids, errs, W = 640, H = 250, yl = 'Error (prediksi − aktual)') {
  const mx = Math.max(...errs.map(Math.abs), 1e-9), yt = ticks(-mx * 1.15, mx * 1.15, 5), Y0 = yt.t[0], Y1 = yt.t[yt.t.length - 1];
  const L = 58, R = 14, T = 20, B = 36, n = ids.length, bw = Math.min(34, (W - L - R) / n * 0.68);
  const Y = v => T + (H - T - B) * (1 - (v - Y0) / (Y1 - Y0)), X = i => L + (W - L - R) * (i + 0.5) / n, yf = tf(yt.step);
  let g = yt.t.map(v => `<line class="gl" x1="${L}" x2="${W - R}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/><text x="${L - 7}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${yf(v)}</text>`).join('');
  g += `<line class="zl" x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}"/>`;
  const skip = Math.ceil(n / 24);
  const body = errs.map((e, i) => `<rect x="${(X(i) - bw / 2).toFixed(1)}" y="${Math.min(Y(0), Y(e)).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, Math.abs(Y(e) - Y(0))).toFixed(1)}" rx="2" fill="${e > 0 ? '#94452A' : '#5B6B8C'}" fill-opacity=".85"><title>${esc(ids[i])}: ${e >= 0 ? '+' : ''}${e.toFixed(2)}</title></rect>${n <= 16 ? `<text class="pt" x="${X(i).toFixed(1)}" y="${(e >= 0 ? Y(e) - 4 : Y(e) + 12).toFixed(1)}" text-anchor="middle">${e >= 0 ? '+' : ''}${e.toFixed(1)}</text>` : ''}${i % skip === 0 ? `<text x="${X(i).toFixed(1)}" y="${H - B + 16}" text-anchor="middle">${esc(ids[i])}</text>` : ''}`).join('');
  return `<svg class="ch" viewBox="0 0 ${W} ${H}" role="img"><text x="4" y="12">${esc(yl)}</text>${g}${body}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Batch (sesi)</text></svg>`;
}

function errStats(a, p) {
  const n = a.length, e = p.map((v, i) => v - a[i]), sum = x => x.reduce((s, v) => s + v, 0);
  const m = sum(a) / n, sst = sum(a.map(v => (v - m) ** 2)), sse = sum(e.map(v => v * v));
  return {n, mae: sum(e.map(Math.abs)) / n, rmse: Math.sqrt(sse / n), bias: sum(e) / n, maxe: Math.max(...e.map(Math.abs)), r2: n > 1 && sst > 1e-12 ? 1 - sse / sst : null};
}
const addMin = (s, m) => { const d = new Date(s.replace(' ', 'T').slice(0, 16) + ':00Z'); d.setUTCMinutes(d.getUTCMinutes() + m); const p = n => String(n).padStart(2, '0'); return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`; };
const unitTxt = c => UNIT[c] ? ` (${UNIT[c]})` : '';

function btView(root, bt, opt = {}) {
  const models = Object.keys(bt.pred), def = models.includes(opt.model) ? opt.model : (models.find(m => m !== 'Persistence') || models[0]);
  const optsHtml = (list, sel) => list.map(([v, l]) => `<option value="${esc(v)}"${String(v) === String(sel) ? ' selected' : ''}>${esc(l)}</option>`).join('');
  root.innerHTML = `${opt.note ? `<p class="mu">${opt.note}</p>` : ''}<div class="row" style="margin:0 0 12px"><label>Model<select class="bm">${optsHtml(models.map(m => [m, m]), def)}</select></label><label>Kanal<select class="bc">${optsHtml(bt.channels.map((c, i) => [i, NAME[c] + unitTxt(c)]), 0)}</select></label><label>Horizon<select class="bh">${optsHtml(bt.horizons.map((h, i) => [i, fmtMin(h)]), 0)}</select></label></div><div class="bo"></div>`;
  const out = root.querySelector('.bo'), val = c => root.querySelector(c).value, W = bt.time.length;
  const draw = () => {
    const m = val('.bm'), c = +val('.bc'), hi = +val('.bh'), ch = bt.channels[c];
    const A = (w, h) => bt.actual[w][c][h], P = (mm, w, h) => bt.pred[mm][w][c][h];
    const ok = h => [...Array(W).keys()].filter(w => isNum(A(w, h)) && isNum(P(m, w, h)));
    const hrows = bt.horizons.map((h, i) => {
      const ws = ok(i), a = ws.map(w => A(w, i)), s = ws.length ? errStats(a, ws.map(w => P(m, w, i))) : null;
      const row = {horizon_min: h, windows: ws.length, MAE: s && s.mae, RMSE: s && s.rmse, bias: s && s.bias, max_abs_error: s && s.maxe};
      if (m !== 'Persistence' && bt.pred.Persistence) { const sp = ws.length ? errStats(a, ws.map(w => P('Persistence', w, i))) : null; row.persistence_MAE = sp && sp.mae; row.skill_vs_persistence = s && sp ? 1 - s.mae / Math.max(sp.mae, 1e-12) : null; }
      return row;
    });
    const ws = ok(hi), rows = ws.map(w => { const a = A(w, hi), p = P(m, w, hi), e = p - a; return {origin_time: bt.time[w], forecast_time: addMin(bt.time[w], bt.horizons[hi]), actual: a, predicted: p, error: e, abs_error: Math.abs(e), error_pct: Math.abs(a) > 1e-9 ? e / Math.abs(a) * 100 : null}; });
    const all = () => { const o = []; bt.channels.forEach((cc, ci) => bt.horizons.forEach((h, i) => { for (let w = 0; w < W; w++) { const a = bt.actual[w][ci][i], p = bt.pred[m][w][ci][i]; if (isNum(a) && isNum(p)) o.push({channel: NAME[cc], horizon_min: h, origin_time: bt.time[w], forecast_time: addMin(bt.time[w], h), actual: a, predicted: p, error: p - a, abs_error: Math.abs(p - a)}); } })); return o; };
    const tip = r => `${r.forecast_time}: aktual ${r.actual.toFixed(3)}, prediksi ${r.predicted.toFixed(3)}, error ${r.error >= 0 ? '+' : ''}${r.error.toFixed(3)}`;
    const idx = rows.map((_, i) => i + 1), tips = rows.map(tip), yl = NAME[ch] + unitTxt(ch);
    out.innerHTML = `<h3>Error per Horizon</h3><p class="mu">${esc(NAME[ch])}, ${esc(m)}. Error dalam ${UNIT[ch] ? esc(UNIT[ch]) : 'satuan sensor'}; error = prediksi − aktual.</p>${tbl(hrows)}
<h3>Aktual vs Prediksi pada ${fmtMin(bt.horizons[hi])} ke Depan</h3>${ws.length ? `<div class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"><div>${legend([['Aktual', '#7A4326'], ['Prediksi', '#46633C']])}${plot([{x: idx, y: rows.map(r => r.actual), c: '#7A4326', pts: true, tips}, {x: idx, y: rows.map(r => r.predicted), c: '#46633C', dash: true, pts: true, tips}], 480, 280, 'Jendela (urut waktu)', yl, false, {nx: 7})}</div><div>${legend([['Jendela', '#7A4326'], ['Prediksi sempurna', '#8C8C8C']])}${plot([{x: rows.map(r => r.actual), y: rows.map(r => r.predicted), c: '#7A4326', dots: true, tips}], 480, 280, 'Aktual', 'Prediksi', true)}</div></div>` : '<p class="mu">Tidak ada jendela tersedia.</p>'}
<h3>Aktual, Prediksi, dan Error</h3><p class="mu">Detail tiap jendela pada horizon ini.</p>${tbl(rows, {csv: 'aktual_vs_prediksi.csv', all})}`;
  };
  ['.bm', '.bc', '.bh'].forEach(s => root.querySelector(s).addEventListener('change', draw));
  draw();
}

function render2() {
  const r = R[2], b = r.results.find(x => x.Model === r.best) || r.results[0], tops = r.importance.slice(0, 3).map(x => x.feature.replace('|', ' ')).join(', ');
  $('#out2').hidden = false;

  // Kotak panduan yang sudah dirapikan
  const guide = `<div class="note" style="background:#FCF9F3; border-left-color:var(--accent); margin-bottom:20px; padding:16px 20px;">
    <div style="font:600 15px var(--serif); margin-bottom:8px; color:var(--ink);">💡 Panduan Singkat:</div>
    <p style="margin:0 0 8px 0; line-height:1.6; color:var(--muted);">Di bawah ini adalah ringkasan kecerdasan AI dalam menebak <strong>Skor Cupping (nilai rasa kopi)</strong>. Untuk melihat seberapa akurat tebakannya, silakan klik tombol <strong>👁️ Lihat Hasil Prediksi (Grafik)</strong>.</p>
    <p style="margin:0; line-height:1.6; color:var(--muted);">Ingin tahu <strong>kapan waktu terbaik memanen kopi</strong>? Pastikan Mode 1 sudah selesai, lalu klik tombol hijau <strong>🔗 Gabungkan Mode 1 & 2 (Rekomendasi Panen)</strong> di bawah.</p>
  </div>`;

  $('#sum2').innerHTML = `${guide}
  <p class="mu">Model dilatih dari ${r.n} batch dengan validasi ketat (${r.cv}).</p>
  <div class="lgrid">
    <div class="stat"><span>Algoritma Terpintar</span><b>${r.best}</b><span>Jenis: ${b.Approach === 'Sequence' ? 'Deep Learning (Sekuens)' : b.Approach}</span></div>
    <div class="stat"><span>Tingkat Kesalahan AI</span><b>${b.MAE.toFixed(2)} poin</b><span>Rata-rata selisih tebakan dengan skor asli (RMSE ${b.RMSE.toFixed(2)})</span></div>
    <div class="stat"><span>Akurasi Penilaian</span><b>${(b.Acc_good * 100).toFixed(0)}%</b><span>Akurasi menebak "Layak" atau "Di Bawah Standar" (ambang batas: ${r.settings.threshold})</span></div>
    <div class="stat"><span>Faktor Penentu Rasa</span><b style="font-size:15px">${tops || 'Tidak ada'}</b><span>Sensor yang paling mempengaruhi hasil akhir</span></div>
  </div>
  ${r.n < 15 ? '<p class="mu" style="margin-top:12px;color:var(--poor)">Catatan: Batch masih sedikit (' + r.n + '). Disarankan minimal 20 batch agar tebakan AI makin jitu.</p>' : ''}`;

  const pb = r.per_batch, bm = r.best, tip = pb.map(x => `Batch ${x.session}: aktual ${x.actual}, prediksi ${x[bm].toFixed(2)}`);
  $('#viz2').innerHTML = `<p class="mu">Kiri: perbandingan Skor Asli vs Skor Tebakan AI (${esc(bm)}). Semakin dekat ke garis putus-putus, semakin akurat. Kanan: rata-rata error tiap model.</p><div class="lgrid" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"><div>${plot([{x: pb.map(x => x.actual), y: pb.map(x => x[bm]), c: '#7A4326', dots: true, labels: pb.map(x => x.session), tips: tip}], 480, 300, 'Skor Asli', 'Skor Prediksi AI', true)}</div><div>${bars(r.results.map(x => ({Model: x.Model, MAE: x.MAE})), 'Model', 'MAE')}</div></div><h3 style="margin:22px 0 4px">Aktual vs Prediksi (${esc(bm)})</h3><p class="mu">Hasil uji silang. Error = prediksi − aktual, dalam poin skor.</p>${tbl(pb.map(x => ({session: x.session, actual: x.actual, predicted: x[bm], error: x[bm] - x.actual, abs_error: Math.abs(x[bm] - x.actual), error_pct: (x[bm] - x.actual) / x.actual * 100})), {csv: 'aktual_vs_prediksi.csv'})}`;
  
  detail2(); predPanel(2);
  if (!QUIET) $('#out2').scrollIntoView({behavior: 'smooth'});
}

document.querySelectorAll('[data-t]').forEach(b => b.addEventListener('click', () => {
  const el = $('#' + b.dataset.t);
  el.hidden = !el.hidden;
}));

document.querySelectorAll('[data-merge]').forEach(b => b.addEventListener('click', async () => {
  const out = $('#mergeOut'), fd = new FormData(), tp = $('#tph').value;
  if (tp) fd.append('target_ph', tp);
  out.hidden = false; out.className = 'lc adv';
  out.innerHTML = '<h3>Menggabungkan Model</h3><div class="bar"><div style="width:100%"></div></div>';
  let r, j;
  try {
    r = await fetch('/api/lab/merge', {method: 'POST', body: fd});
    const txt = await r.text();
    try { j = JSON.parse(txt); } catch (e) { j = {error: `Server mengembalikan ${r.status}: ${txt.slice(0, 200) || 'respons kosong'}`}; }
  } catch (e) {
    r = {ok: false};
    j = {error: 'Tidak dapat menghubungi server. Server mungkin kehabisan memori dan restart.'};
  }
  if (!r.ok) { out.innerHTML = `<h3>Model Belum Bisa Digabungkan</h3><p class="err">${j.error}</p>`; out.scrollIntoView({behavior: 'smooth'}); return; }
  out.classList.add('good');
  const a = j.advice, i = a.best_min / 10, f = j.future;
  out.innerHTML = `<h3>Rekomendasi Gabungan</h3><div class="big" style="font:600 22px var(--serif);margin:6px 0">Waktu Henti yang Disarankan: sekitar ${fmtMin(a.best_min)} lagi, prediksi skor ${a.best_score.toFixed(1)} (${a.best_score >= 80 ? 'layak' : 'di bawah standar'})</div><p class="mu">Prakiraan pada waktu tersebut: pH ${num('ph', f.ph[i])}, CO₂ ${num('co2', f.co2[i])} ppm, suhu proses ${num('temp_process', f.temp_process[i])} °C. Estimasi ini andal hanya jika durasi saat ini dekat dengan durasi batch pelatihan Mode 2.</p>${plot([{x: f.minutes.slice(1).map(m => m / 60), y: a.scores, c: '#46633C'}], 900, 280, 'Jam dari Sekarang Jika Fermentasi Dihentikan', 'Prediksi Skor Cupping')}`;
  out.scrollIntoView({behavior: 'smooth'});
}));
})();