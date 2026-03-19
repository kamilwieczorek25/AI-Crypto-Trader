/* ─── Config ─── */
const API_BASE = `http://${location.hostname}:9000`;
const WS_URL   = `ws://${location.hostname}:9000/ws`;

/* ─── State ─── */
let ws = null;
let wsReconnectTimer = null;
let wsReconnectDelay = 1000;  // exponential backoff starting at 1s
let currentTf = '15m';
let tradePage = 1;
let priceChart = null, rsiChart = null, macdChart = null;
let growthChart = null, allocChart = null, pnlChart = null;
let knownSymbols = new Set();

const PALETTE = ['#6366f1','#22c55e','#f59e0b','#3b82f6','#ef4444','#8b5cf6','#14b8a6','#f97316','#ec4899','#06b6d4'];

/* ─── DOM helpers ─── */
const $ = id => document.getElementById(id);
const fmt = (n, dec=2) => n == null ? '—' : Number(n).toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec});
const fmtPrice = n => n == null ? '—' : `$${fmt(n, n < 1 ? 6 : 2)}`;
const fmtTime = s => s ? new Date(s).toLocaleString() : '—';

function timeAgo(dateStr) {
  if (!dateStr) return '—';
  const diff = (Date.now() - new Date(dateStr)) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

function pnlClass(v) { return v == null ? '' : v >= 0 ? 'pnl-pos' : 'pnl-neg'; }
function pnlSign(v) { return v == null ? '—' : (v >= 0 ? '+' : '') + fmt(v); }

/* ─── WebSocket ─── */
function connectWs() {
  if (ws && ws.readyState < 2) return;
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    $('ws-dot').className = 'dot dot-green';
    $('footer-msg').textContent = 'Connected';
    clearTimeout(wsReconnectTimer);
    wsReconnectDelay = 1000;  // reset backoff on successful connection
  };

  ws.onclose = () => {
    $('ws-dot').className = 'dot dot-red';
    $('footer-msg').textContent = `Disconnected — reconnecting in ${Math.round(wsReconnectDelay/1000)}s…`;
    wsReconnectTimer = setTimeout(connectWs, wsReconnectDelay);
    wsReconnectDelay = Math.min(wsReconnectDelay * 2, 30000);  // cap at 30s
  };

  ws.onerror = () => ws.close();

  ws.onmessage = evt => {
    try {
      const msg = JSON.parse(evt.data);
      handleEvent(msg.type, msg.data);
    } catch (_) {}
  };
}

function handleEvent(type, data) {
  switch (type) {
    case 'PORTFOLIO_UPDATE':
      renderPortfolio(data);
      loadGrowthChart();
      break;
    case 'CLAUDE_DECISION':
      renderDecision(data);
      loadDecisionFeed();
      break;
    case 'TRADE_EXECUTED':
      loadTrades(0);
      loadPnlChart();
      loadAnalytics();
      break;
    case 'BOT_STATUS':        renderBotStatus(data); break;
    case 'PRICE_TICK':        handlePriceTick(data); break;
    case 'MARKET_DATA_UPDATE': handleMarketUpdate(data); break;
    case 'ERROR':             $('footer-msg').textContent = `Error: ${data.message}`; break;
  }
}

/* ─── Portfolio ─── */
function renderPortfolio(p) {
  $('stat-total').textContent     = `$${fmt(p.total_value_usdt)}`;
  $('stat-cash').textContent      = `$${fmt(p.cash_usdt)}`;
  $('stat-positions').textContent = `$${fmt(p.positions_value_usdt)}`;
  $('stat-num-pos').textContent   = p.num_open_positions;

  const pnlEl = $('stat-pnl');
  pnlEl.textContent = `${p.total_pnl_usdt >= 0 ? '+' : ''}$${fmt(p.total_pnl_usdt)} (${pnlSign(p.total_pnl_pct)}%)`;
  pnlEl.className = 'stat-value ' + pnlClass(p.total_pnl_usdt);

  renderPositions(p.positions || []);
  loadAllocChart(p);
}

function renderPositions(positions) {
  const tbody = $('positions-body');
  if (!positions.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-msg">No open positions</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => `
    <tr>
      <td><strong>${p.symbol}</strong></td>
      <td>${fmt(p.quantity, 6)}</td>
      <td>${fmtPrice(p.avg_entry_price)}</td>
      <td>${fmtPrice(p.current_price)}</td>
      <td>$${fmt(p.value_usdt)}</td>
      <td class="${pnlClass(p.pnl_usdt)}">${pnlSign(p.pnl_usdt)}</td>
      <td class="${pnlClass(p.pnl_pct)}">${pnlSign(p.pnl_pct)}%</td>
      <td>${fmtPrice(p.stop_loss_price)}</td>
      <td>${fmtPrice(p.take_profit_price)}</td>
    </tr>`).join('');
}

/* ─── Claude Decision ─── */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

function renderDecision(d) {
  const actionClass = {BUY:'action-buy', SELL:'action-sell', HOLD:'action-hold'}[d.action] || '';
  const confPct = Math.round((d.confidence || 0) * 100);
  const signals = (d.primary_signals || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');
  const risks   = (d.risk_factors || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');

  $('decision-content').innerHTML = `
    <div class="decision-box">
      <div>
        <div class="decision-action ${actionClass}">${escapeHtml(d.action)}</div>
        <div style="margin-top:8px;font-size:13px;">
          <strong>${escapeHtml(d.symbol)}</strong> &nbsp;·&nbsp; ${escapeHtml(d.timeframe)} &nbsp;·&nbsp; ${fmt(d.quantity_pct,1)}% of portfolio
        </div>
        <div class="confidence-bar-wrap" style="margin-top:10px;">
          <span style="font-size:11px;color:var(--muted);">Confidence</span>
          <div class="confidence-bg"><div class="confidence-bar" style="width:${confPct}%"></div></div>
          <span style="font-size:12px;">${confPct}%</span>
        </div>
      </div>
      <div class="decision-meta">
        <span>Primary signals</span>
        <ul class="signal-list">${signals || '<li>—</li>'}</ul>
        <span style="margin-top:8px;">Risk factors</span>
        <ul class="signal-list">${risks || '<li>—</li>'}</ul>
      </div>
      <div>
        <span style="font-size:11px;color:var(--muted);">Reasoning</span>
        <p class="reasoning-text">${escapeHtml(d.reasoning) || '—'}</p>
      </div>
    </div>`;
}

/* ─── Bot Status ─── */
function renderBotStatus(s) {
  $('stat-status').textContent = s.running ? '▶ RUNNING' : '■ STOPPED';
  $('stat-status').style.color = s.running ? 'var(--green)' : 'var(--muted)';
  $('btn-start').disabled = s.running;
  $('btn-stop').disabled  = !s.running;

  if (s.next_cycle_in_seconds != null && s.running) {
    $('next-cycle').textContent = `Next cycle in ${s.next_cycle_in_seconds}s`;
  } else {
    $('next-cycle').textContent = '';
  }

  // Sync mode buttons
  const mode = (s.mode || 'demo').toLowerCase();
  $('btn-demo').classList.toggle('active', mode === 'demo');
  $('btn-real').classList.toggle('active', mode === 'real');

  // Sync risk profile buttons
  if (s.risk_profile) {
    syncRiskButtons(s.risk_profile.key || 'balanced');
  }
}

/* ─── Risk Profile ─── */
const RISK_PROFILES = ['conservative', 'balanced', 'aggressive', 'fast_profit'];

function syncRiskButtons(activeKey) {
  RISK_PROFILES.forEach(key => {
    const btn = $(`risk-${key}`);
    if (!btn) return;
    RISK_PROFILES.forEach(k => btn.classList.remove(`active-${k}`));
    if (key === activeKey) btn.classList.add(`active-${key}`);
  });
}

async function setRiskProfile(profile) {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/risk-profile`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({profile}),
    });
    if (!resp.ok) { const e = await resp.json(); alert(e.detail || 'Failed'); return; }
    const data = await resp.json();
    syncRiskButtons(data.risk_profile.key);
  } catch (e) { alert('Error: ' + e.message); }
}

/* ─── Price tick ─── */
function handlePriceTick(data) {
  loadPortfolio();
}

/* ─── Market update — populate symbol selector ─── */
function handleMarketUpdate(data) {
  const sel = $('chart-symbol');
  Object.keys(data).forEach(sym => {
    if (!knownSymbols.has(sym)) {
      knownSymbols.add(sym);
      const opt = document.createElement('option');
      opt.value = sym;
      opt.textContent = sym;
      sel.appendChild(opt);
    }
  });
}

/* ─── Trade history ─── */
async function loadTrades(delta) {
  tradePage = Math.max(1, tradePage + delta);
  const resp = await fetch(`${API_BASE}/api/trades?page=${tradePage}&page_size=20`);
  const data = await resp.json();
  $('page-info').textContent = `Page ${tradePage}`;
  $('btn-prev').disabled = tradePage <= 1;

  const tbody = $('trades-body');
  const items = data.items || [];
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">No trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = items.map(t => `
    <tr>
      <td>${fmtTime(t.created_at)}</td>
      <td><strong>${t.symbol}</strong></td>
      <td class="${t.direction==='BUY'?'dir-buy':'dir-sell'}">${t.direction}</td>
      <td><span class="badge-${t.mode}">${t.mode.toUpperCase()}</span></td>
      <td>${fmt(t.quantity, 6)}</td>
      <td>${fmtPrice(t.price)}</td>
      <td class="${pnlClass(t.pnl_usdt)}">${pnlSign(t.pnl_usdt)}</td>
      <td class="${pnlClass(t.pnl_pct)}">${pnlSign(t.pnl_pct)}%</td>
    </tr>`).join('');
}

/* ─── Chart ─── */
function selectTf(btn, tf) {
  document.querySelectorAll('.btn-tf').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentTf = tf;
  loadChart();
}

async function loadChart() {
  const sym = $('chart-symbol').value;
  if (!sym) return;
  const symEncoded = sym.replace('/', '_');
  const resp = await fetch(`${API_BASE}/api/ohlcv/${symEncoded}/${currentTf}`);
  if (!resp.ok) return;
  const candles = await resp.json();
  if (!candles.length) return;

  const labels = candles.map(c => new Date(c.t).toLocaleTimeString());
  const closes = candles.map(c => c.c);
  const rsiValues = calcRSI(closes, 14);
  const macdData  = calcMACD(closes, 12, 26, 9);

  renderPriceChart(labels, closes);
  renderRsiChart(labels, rsiValues);
  renderMacdChart(labels, macdData);
}

function renderPriceChart(labels, closes) {
  if (priceChart) priceChart.destroy();
  priceChart = new Chart($('price-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Price',
        data: closes,
        borderColor: '#6366f1',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
        fill: false,
      }]
    },
    options: chartOptions('Price (USDT)'),
  });
}

function renderRsiChart(labels, rsi) {
  if (rsiChart) rsiChart.destroy();
  rsiChart = new Chart($('rsi-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'RSI 14', data: rsi, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, fill: false },
      ]
    },
    options: {
      ...chartOptions('RSI'),
      plugins: { legend: { display: true, labels: { color: '#8b8fa8', font: { size: 10 } } } },
    },
  });
}

function renderMacdChart(labels, macd) {
  if (macdChart) macdChart.destroy();
  macdChart = new Chart($('macd-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'MACD Histogram',
        data: macd.hist,
        backgroundColor: macd.hist.map(v => v >= 0 ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)'),
        borderRadius: 1,
      }]
    },
    options: chartOptions('MACD Hist'),
  });
}

function chartOptions(label) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { display: false }, tooltip: { mode: 'index' } },
    scales: {
      x: { ticks: { color: '#8b8fa8', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#1e2035' } },
      y: { ticks: { color: '#8b8fa8', font: { size: 10 } }, grid: { color: '#1e2035' } },
    },
  };
}

/* ─── RSI (simple EMA-based) ─── */
function calcRSI(closes, period=14) {
  const result = new Array(closes.length).fill(null);
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i-1];
    if (d >= 0) gains += d; else losses -= d;
  }
  let avgGain = gains / period, avgLoss = losses / period;
  result[period] = 100 - (100 / (1 + avgGain / (avgLoss || 0.001)));
  for (let i = period + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i-1];
    avgGain = (avgGain * (period-1) + Math.max(d, 0)) / period;
    avgLoss = (avgLoss * (period-1) + Math.max(-d, 0)) / period;
    result[i] = 100 - (100 / (1 + avgGain / (avgLoss || 0.001)));
  }
  return result;
}

/* ─── MACD ─── */
function calcMACD(closes, fast=12, slow=26, signal=9) {
  const ema = (data, period) => {
    const k = 2 / (period + 1);
    const result = new Array(data.length).fill(null);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) { sum += data[i]; continue; }
      if (i === period - 1) { sum += data[i]; result[i] = sum / period; continue; }
      result[i] = data[i] * k + result[i-1] * (1 - k);
    }
    return result;
  };
  const fastEma = ema(closes, fast);
  const slowEma = ema(closes, slow);
  const macdLine = closes.map((_, i) =>
    fastEma[i] != null && slowEma[i] != null ? fastEma[i] - slowEma[i] : null);
  const sigLine = ema(macdLine.filter(v => v != null), signal);
  let sigIdx = 0;
  const hist = macdLine.map(m => {
    if (m == null) return null;
    const s = sigLine[sigIdx++];
    return s != null ? m - s : null;
  });
  return { macd: macdLine, signal: sigLine, hist };
}

/* ─── Portfolio Growth Chart ─── */
async function loadGrowthChart() {
  try {
    const resp = await fetch(`${API_BASE}/api/snapshots?limit=200`);
    const snaps = await resp.json();
    if (!snaps.length) return;

    const labels = snaps.map(s => {
      const d = new Date(s.created_at);
      return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    });
    const values = snaps.map(s => s.total_value_usdt);

    const canvas = $('growth-chart');
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(99,102,241,0.35)');
    gradient.addColorStop(1, 'rgba(99,102,241,0.02)');

    if (growthChart) growthChart.destroy();
    growthChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Portfolio Value',
          data: values,
          borderColor: '#6366f1',
          backgroundColor: gradient,
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` $${fmt(ctx.raw)}` } },
        },
        scales: {
          x: { ticks: { color: '#8b8fa8', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#1e2035' } },
          y: { ticks: { color: '#8b8fa8', font: { size: 10 }, callback: v => `$${fmt(v, 0)}` }, grid: { color: '#1e2035' } },
        },
      }
    });
  } catch (_) {}
}

/* ─── Allocation Donut ─── */
function loadAllocChart(portfolio) {
  if (!portfolio) return;

  const labels = ['Cash'];
  const values = [Math.max(portfolio.cash_usdt || 0, 0)];
  const colors = ['#4b5563'];

  (portfolio.positions || []).forEach((p, i) => {
    labels.push(p.symbol.replace('/USDT', ''));
    values.push(Math.max(p.value_usdt || 0, 0));
    colors.push(PALETTE[i % PALETTE.length]);
  });

  const total = values.reduce((a, b) => a + b, 0) || 1;

  if (allocChart) allocChart.destroy();
  allocChart = new Chart($('alloc-chart'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: '#161824',
        borderWidth: 3,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` $${fmt(ctx.raw)} (${(ctx.raw / total * 100).toFixed(1)}%)`
          }
        }
      }
    }
  });

  const legend = $('alloc-legend');
  if (legend) {
    legend.innerHTML = labels.map((l, i) =>
      `<div class="alloc-legend-item"><div class="alloc-dot" style="background:${colors[i]}"></div>${l} ${(values[i] / total * 100).toFixed(1)}%</div>`
    ).join('');
  }
}

/* ─── Trade P&L Chart ─── */
async function loadPnlChart() {
  try {
    const resp = await fetch(`${API_BASE}/api/trades?page=1&page_size=100`);
    const data = await resp.json();
    const sells = (data.items || [])
      .filter(t => t.direction === 'SELL' && t.pnl_usdt != null)
      .reverse(); // chronological order

    const canvas = $('pnl-chart');
    if (!canvas) return;

    if (!sells.length) {
      if (pnlChart) { pnlChart.destroy(); pnlChart = null; }
      return;
    }

    const labels = sells.map(t => t.symbol.replace('/USDT', ''));
    const values = sells.map(t => t.pnl_usdt);

    if (pnlChart) pnlChart.destroy();
    pnlChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'P&L (USDT)',
          data: values,
          backgroundColor: values.map(v => v >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'),
          borderColor: values.map(v => v >= 0 ? '#22c55e' : '#ef4444'),
          borderWidth: 1,
          borderRadius: 3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${ctx.raw >= 0 ? '+' : ''}$${fmt(ctx.raw)}` } },
        },
        scales: {
          x: { ticks: { color: '#8b8fa8', font: { size: 10 }, maxRotation: 45 }, grid: { display: false } },
          y: { ticks: { color: '#8b8fa8', font: { size: 10 }, callback: v => `$${fmt(v, 0)}` }, grid: { color: '#1e2035' } },
        }
      }
    });
  } catch (_) {}
}

/* ─── Analytics Stats ─── */
async function loadAnalytics() {
  try {
    const resp = await fetch(`${API_BASE}/api/analytics`);
    if (!resp.ok) return;
    const d = await resp.json();
    const el = $('analytics-stats');
    if (!el) return;

    const wr    = d.closed_trades > 0 ? `${Math.round(d.win_rate * 100)}%` : '—';
    const avgPnl = d.closed_trades > 0 ? `${d.avg_pnl_usdt >= 0 ? '+' : ''}$${fmt(d.avg_pnl_usdt)}` : '—';
    const best  = d.best_pnl_usdt  > 0 ? `+$${fmt(d.best_pnl_usdt)}`              : '—';
    const worst = d.worst_pnl_usdt < 0 ? `-$${fmt(Math.abs(d.worst_pnl_usdt))}` : '—';
    const wrColor = d.win_rate >= 0.5 ? 'var(--green)' : 'var(--red)';

    el.innerHTML = `
      <div class="a-stat"><div class="a-val">${d.closed_trades}</div><div class="a-lbl">Closed</div></div>
      <div class="a-stat"><div class="a-val" style="color:${wrColor}">${wr}</div><div class="a-lbl">Win Rate</div></div>
      <div class="a-stat"><div class="a-val ${d.avg_pnl_usdt >= 0 ? 'pnl-pos' : 'pnl-neg'}">${avgPnl}</div><div class="a-lbl">Avg P&amp;L</div></div>
      <div class="a-stat"><div class="a-val pnl-pos">${best}</div><div class="a-lbl">Best</div></div>
      <div class="a-stat"><div class="a-val pnl-neg">${worst}</div><div class="a-lbl">Worst</div></div>`;
  } catch (_) {}
}

/* ─── AI Decision Feed ─── */
async function loadDecisionFeed() {
  try {
    const resp = await fetch(`${API_BASE}/api/decisions?page=1&page_size=20`);
    const data = await resp.json();
    const feed = $('decision-feed');
    if (!feed) return;

    const items = data.items || [];
    if (!items.length) {
      feed.innerHTML = '<div class="empty-msg">Waiting for decisions…</div>';
      return;
    }

    feed.innerHTML = items.map(d => {
      const confPct = Math.round((d.confidence || 0) * 100);
      const badgeCls = {BUY: 'feed-buy', SELL: 'feed-sell', HOLD: 'feed-hold'}[d.action] || 'feed-hold';
      const signal = (d.primary_signals && d.primary_signals[0]) || (d.reasoning || '').slice(0, 80) || '—';
      const sym = d.symbol || 'MARKET';
      return `
        <div class="feed-item">
          <span class="feed-time">${timeAgo(d.created_at)}</span>
          <span class="feed-badge ${badgeCls}">${d.action}</span>
          <span class="feed-sym">${sym}</span>
          <div style="display:flex;align-items:center;gap:5px;min-width:90px;">
            <div class="feed-conf-bg"><div class="feed-conf-bar" style="width:${confPct}%"></div></div>
            <span style="font-size:10px;color:var(--muted)">${confPct}%</span>
          </div>
          <span class="feed-snippet" title="${(d.primary_signals || []).join(' | ')}">${signal}</span>
        </div>`;
    }).join('');
  } catch (_) {}
}

/* ─── API calls ─── */
async function loadPortfolio() {
  try {
    const resp = await fetch(`${API_BASE}/api/portfolio`);
    const data = await resp.json();
    renderPortfolio(data);
  } catch (_) {}
}

async function setMode(mode) {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/mode`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({mode}),
    });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.detail || 'Mode change failed');
      return;
    }
    $('btn-demo').classList.toggle('active', mode === 'demo');
    $('btn-real').classList.toggle('active', mode === 'real');
  } catch (e) { alert('Error: ' + e.message); }
}

async function startBot() {
  await fetch(`${API_BASE}/api/bot/start`, {method:'POST'});
}

async function stopBot() {
  await fetch(`${API_BASE}/api/bot/stop`, {method:'POST'});
}

async function resetDemo() {
  if (!confirm('Reset demo? This will wipe ALL trade history, decisions, and positions and restore the $10,000 balance.')) return;
  const resp = await fetch(`${API_BASE}/api/bot/reset-demo`, {method:'POST'});
  if (resp.ok) {
    await Promise.all([
      loadPortfolio(), loadTrades(0), syncBotStatus(), loadLastDecision(),
      loadPnlChart(), loadAnalytics(), loadDecisionFeed(), loadGrowthChart(),
    ]);
    $('decision-content').innerHTML = '<div class="decision-empty">History cleared. Start the bot to begin.</div>';
  } else {
    alert('Reset failed');
  }
}

/* ─── Bot status REST poll (survives page refresh) ─── */
async function syncBotStatus() {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/status`);
    const data = await resp.json();
    renderBotStatus(data);
  } catch (_) {}
}

/* ─── Claude session cost ─── */
async function loadCreditBalance() {
  try {
    const resp = await fetch(`${API_BASE}/api/usage`);
    if (!resp.ok) return;
    const data = await resp.json();
    const el = $('stat-credits');
    if (!el) return;
    const cost = Number(data.total_cost_usd || 0);
    el.textContent = `$${cost.toFixed(4)}`;
    el.title =
      `Calls: ${data.total_calls} | ` +
      `Input: ${(data.total_input_tokens/1000).toFixed(1)}k tokens | ` +
      `Output: ${(data.total_output_tokens/1000).toFixed(1)}k tokens`;
  } catch (_) {}
}

/* ─── Last decision REST fetch (survives page refresh) ─── */
async function loadLastDecision() {
  try {
    const resp = await fetch(`${API_BASE}/api/decisions?page=1&page_size=1`);
    const data = await resp.json();
    if (data.items && data.items.length) renderDecision(data.items[0]);
  } catch (_) {}
}

/* ─── Init ─── */
(async () => {
  connectWs();
  // Restore full UI state from REST on every page load (no dependency on WS session)
  await Promise.all([
    loadPortfolio(),
    loadTrades(0),
    syncBotStatus(),
    loadLastDecision(),
    loadCreditBalance(),
    loadGrowthChart(),
    loadPnlChart(),
    loadAnalytics(),
    loadDecisionFeed(),
  ]);

  // Periodic fallback polls (in case WS events are missed)
  setInterval(loadPortfolio, 60_000);
  setInterval(syncBotStatus, 30_000);
  setInterval(loadCreditBalance, 300_000);
  setInterval(loadGrowthChart, 60_000);
  setInterval(loadAnalytics, 60_000);
  setInterval(loadDecisionFeed, 30_000);
})();
