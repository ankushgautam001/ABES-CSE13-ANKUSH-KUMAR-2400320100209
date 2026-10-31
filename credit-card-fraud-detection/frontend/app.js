const $ = id => document.getElementById(id);
const pct = x => (100 * x).toFixed(1) + '%';
const FIGS = [['03_pr_roc_curves.png','PR and ROC curves'],['04_model_comparison.png','Model comparison'],
  ['05_best_model_diagnostics.png','Best model diagnostics'],['06_alert_budget.png','Alert budget'],
  ['01_eda_overview.png','Data overview'],['02_feature_shift.png','Feature shift']];
const ELEVATED = 95;               // risk score from which unflagged transactions are shown as "elevated"
let sample = null;                 // last loaded transaction (keeps hidden columns)

async function api(path, opts) {
  const r = await fetch(path, opts), j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function band(r) { return r.flagged ? 'high' : r.risk_score >= ELEVATED ? 'elev' : 'low'; }
const bandText = {high: 'FLAGGED: send to review', elev: 'Elevated risk: monitor', low: 'Looks normal'};

async function init() {
  try {
    const h = await api('/api/health');
    $('status').textContent = 'API online · ' + h.data_source; $('status').className = 'pill ok';
    const m = await api('/api/metrics'), best = m.results[m.best_anomaly_model];
    $('kpis').innerHTML = [['Best anomaly model', m.best_anomaly_model.replace(' (Gaussian)', '')],
      ['ROC-AUC', best.roc_auc.toFixed(3)], ['PR-AUC', best.pr_auc.toFixed(3)],
      ['Precision', pct(best.precision)], ['Recall', pct(best.recall)], ['Loss reduction', best.saving_pct.toFixed(0) + '%']]
      .map(([k, v]) => `<div class="kpi"><b>${esc(v)}</b><span>${k}</span></div>`).join('');
    const rows = Object.entries(m.results).sort((a, b) => b[1].pr_auc - a[1].pr_auc);
    $('cmp').innerHTML = '<tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Frauds caught</th><th>False alarms</th></tr>' +
      rows.map(([n, r]) => `<tr><td>${esc(n)}</td><td>${r.roc_auc.toFixed(3)}</td><td>${r.pr_auc.toFixed(3)}</td><td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${r.f1.toFixed(3)}</td><td>${r.tp} / ${r.tp + r.fn}</td><td>${r.fp}</td></tr>`).join('');
    $('figs').innerHTML = FIGS.map(([f, t]) => `<figure style="margin:0"><img loading="lazy" src="/figures/${f}" alt="${t}"><figcaption class="muted">${t}</figcaption></figure>`).join('');
  } catch (e) { $('status').textContent = 'API offline'; $('status').className = 'pill err'; }
}

function fill(row) {
  sample = row;
  $('fAmount').value = row.Amount;
  $('fHour').value = Math.floor((row.Time % 86400) / 3600);
  const v = {}; for (let i = 1; i <= 28; i++) v['V' + i] = row['V' + i];
  $('fV').value = JSON.stringify(v);
  $('single').hidden = true;
  $('single').dataset.truth = 'Class' in row ? row.Class : '';
}

async function scoreOne() {
  try {
    const v = JSON.parse($('fV').value || '{}');
    const base = sample ? sample.Time - (sample.Time % 86400) : 0;
    const rec = {...v, Amount: $('fAmount').value, Time: base + Number($('fHour').value) * 3600 + 1800};
    const res = await api('/api/score', {method: 'POST', body: JSON.stringify(rec)});
    const r = res.results[0], b = band(r), box = $('single');
    const truth = box.dataset.truth === '' ? '' : `<p class="muted">Ground truth for this sample: <b>${box.dataset.truth == 1 ? 'fraud' : 'legitimate'}</b></p>`;
    box.innerHTML = `<div class="verdict ${b}">${bandText[b]}</div>
      <div class="meter"><i style="width:${r.risk_score}%"></i></div>
      <p>Risk score <b>${r.risk_score}</b> / 100 · anomaly score ${r.anomaly_score} · model: ${esc(res.model)}</p>
      <p>Most unusual features: <b>${esc(r.top_reasons)}</b></p>${truth}`;
    box.hidden = false;
  } catch (e) { alert('Could not score: ' + e.message); }
}

function showBatch(res) {
  const rs = res.results, hasY = rs.length && 'Class' in rs[0];
  const flagged = rs.filter(r => r.flagged);
  let k = [['Transactions', rs.length], ['Flagged', flagged.length], ['Alert rate', pct(flagged.length / rs.length)]];
  if (hasY) {
    const fr = rs.filter(r => r.Class == 1).length, tp = flagged.filter(r => r.Class == 1).length;
    k.push(['Frauds caught', `${tp} / ${fr}`], ['Precision of flags', flagged.length ? pct(tp / flagged.length) : 'n/a']);
  }
  $('batchSummary').innerHTML = k.map(([a, b]) => `<div class="kpi"><b>${b}</b><span>${a}</span></div>`).join('');
  const sorted = [...rs].sort((a, b) => b.risk_score - a.risk_score).slice(0, 300);
  $('batch').innerHTML = '<tr><th>#</th><th>Amount</th><th>Risk</th><th>Status</th><th>Top reasons</th>' + (hasY ? '<th>Actual</th>' : '') + '</tr>' +
    sorted.map((r, i) => `<tr class="${band(r) === 'high' ? 'flag' : band(r) === 'elev' ? 'elev' : ''}"><td>${i + 1}</td><td>${r.Amount}</td><td>${r.risk_score}</td><td>${r.flagged ? 'Flagged' : r.risk_score >= ELEVATED ? 'Elevated' : 'OK'}</td><td>${esc(r.top_reasons)}</td>${hasY ? `<td>${r.Class == 1 ? 'Fraud' : 'Legit'}</td>` : ''}</tr>`).join('');
}

$('btnRand').onclick = async () => fill(await api('/api/sample'));
$('btnFraud').onclick = async () => fill(await api('/api/sample_fraud'));
$('btnScore').onclick = scoreOne;
$('btnDemo').onclick = async () => {
  try { showBatch(await api('/api/score', {method: 'POST', body: JSON.stringify(await api('/api/sample_batch'))})); }
  catch (e) { alert(e.message); }
};
$('file').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try { showBatch(await api('/api/score_csv', {method: 'POST', body: await f.text()})); }
  catch (err) { alert('Upload failed: ' + err.message); }
};
init(); api('/api/sample').then(fill).catch(() => {});
