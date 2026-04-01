/* ═══════════════════════════════════════════════════════════
   AI Crypto Trader — Dashboard v2
   ═══════════════════════════════════════════════════════════ */

/* ─── Config ─── */
const API_BASE = `http://${location.hostname}:9000`;
const WS_URL   = `ws://${location.hostname}:9000/ws`;

/* ─── State ─── */
let ws = null;
let wsReconnectTimer = null;
let wsReconnectDelay = 1000;
let currentTf = '15m';
let tradePage = 1;
let priceChart = null, rsiChart = null, macdChart = null;
let growthChart = null, allocChart = null, pnlChart = null;
let knownSymbols = new Set();

const PALETTE = ['#7c4dff','#00e676','#ffd740','#448aff','#ff5252','#b388ff','#18ffff','#ff9100','#ec4899','#06b6d4'];

/* ─── Button tooltips (data-tip attribute → styled popup) ─── */
(function initTooltips() {
  const box = document.getElementById('btn-tooltip');
  if (!box) return;
  let hideTimer = null;
  let _lastTapped = null;

  // ── Mouse (desktop) ──────────────────────────────────────────
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) return;
    clearTimeout(hideTimer);
    box.innerHTML = el.dataset.tip;
    box.classList.add('visible');
    positionTooltip(el);
  });

  document.addEventListener('mouseout', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) return;
    hideTimer = setTimeout(() => box.classList.remove('visible'), 80);
  });

  document.addEventListener('mousemove', e => {
    if (!box.classList.contains('visible')) return;
    const el = e.target.closest('[data-tip]');
    if (el) positionTooltip(el);
  });

  // ── Touch (mobile): tap the button to show/hide tooltip ──────
  document.addEventListener('touchstart', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) {
      // Tap outside any data-tip element → dismiss
      if (box.classList.contains('visible')) {
        box.classList.remove('visible');
        _lastTapped = null;
      }
      return;
    }
    e.preventDefault(); // prevent ghost click
    if (_lastTapped === el && box.classList.contains('visible')) {
      // Second tap on same element → dismiss and fire click
      box.classList.remove('visible');
      _lastTapped = null;
      el.click();
    } else {
      // First tap → show tooltip
      box.innerHTML = el.dataset.tip;
      box.classList.add('visible');
      positionTooltip(el);
      _lastTapped = el;
    }
  }, { passive: false });

  function positionTooltip(el) {
    const r = el.getBoundingClientRect();
    const tw = Math.min(310, window.innerWidth - 16);
    box.style.maxWidth = tw + 'px';
    // Try below first, flip above if not enough space
    let top = r.bottom + 8;
    if (top + 160 > window.innerHeight) top = Math.max(8, r.top - 8 - box.offsetHeight);
    let left = r.left + r.width / 2 - tw / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    box.style.top  = top  + 'px';
    box.style.left = left + 'px';
  }
})();

/* ─── DOM helpers ─── */
const $ = id => document.getElementById(id);
const fmt = (n, dec=2) => n == null ? '—' : Number(n).toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec});
const fmtPrice = n => n == null ? '—' : `$${fmt(n, n < 1 ? 6 : 2)}`;
const fmtTime = s => s ? new Date(s.endsWith('Z') ? s : s + 'Z').toLocaleString() : '—';

function timeAgo(dateStr) {
  if (!dateStr) return '—';
  const utcStr = dateStr.endsWith('Z') ? dateStr : dateStr + 'Z';
  return new Date(utcStr).toLocaleString('pl-PL', { timeZone: 'Europe/Warsaw', hour12: false });
}

function holdDuration(openedAt) {
  if (!openedAt) return '—';
  const utcStr = openedAt.endsWith('Z') ? openedAt : openedAt + 'Z';
  const ms = Date.now() - new Date(utcStr).getTime();
  if (ms < 0) return '—';
  const mins = Math.floor(ms / 60000);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ${mins % 60}m`;
  const days = Math.floor(hrs / 24);
  return `${days}d ${hrs % 24}h`;
}

function pnlClass(v) { return v == null ? '' : v >= 0 ? 'pnl-pos' : 'pnl-neg'; }
function pnlSign(v) { return v == null ? '—' : (v >= 0 ? '+' : '') + fmt(v); }

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

/* ═══════════════════════════════════════════════════════════
   WebSocket
   ═══════════════════════════════════════════════════════════ */
function connectWs() {
  if (ws && ws.readyState < 2) return;
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    const dot = $('ws-dot');
    const label = $('ws-label');
    if (dot) { dot.className = 'ws-dot connected'; }
    if (label) { label.textContent = 'Live'; label.style.color = 'var(--green)'; }
    const fm = $('footer-msg');
    if (fm) fm.textContent = 'Connected';
    clearTimeout(wsReconnectTimer);
    wsReconnectDelay = 1000;
  };

  ws.onclose = () => {
    const dot = $('ws-dot');
    const label = $('ws-label');
    if (dot) { dot.className = 'ws-dot disconnected'; }
    if (label) { label.textContent = 'Offline'; label.style.color = ''; }
    const fm = $('footer-msg');
    if (fm) fm.textContent = `Disconnected — reconnecting in ${Math.round(wsReconnectDelay/1000)}s…`;
    wsReconnectTimer = setTimeout(connectWs, wsReconnectDelay);
    wsReconnectDelay = Math.min(wsReconnectDelay * 2, 30000);
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
    case 'ERROR':
      const fm = $('footer-msg');
      if (fm) fm.textContent = `Error: ${data.message}`;
      break;
  }
}

/* ═══════════════════════════════════════════════════════════
   Portfolio
   ═══════════════════════════════════════════════════════════ */
function renderPortfolio(p) {
  const setEl = (id, val) => { const el = $(id); if (el) el.textContent = val; };

  setEl('stat-total', `$${fmt(p.total_value_usdt)}`);
  setEl('stat-cash', `$${fmt(p.cash_usdt)}`);
  setEl('stat-positions', `$${fmt(p.positions_value_usdt)}`);
  setEl('stat-num-pos', p.num_open_positions);

  const posCountEl = $('positions-count');
  if (posCountEl) posCountEl.textContent = p.num_open_positions;

  const pnlEl = $('stat-pnl');
  if (pnlEl) {
    pnlEl.textContent = `${p.total_pnl_usdt >= 0 ? '+' : ''}$${fmt(p.total_pnl_usdt)} (${pnlSign(p.total_pnl_pct)}%)`;
    pnlEl.className = 'kpi-value ' + pnlClass(p.total_pnl_usdt);
  }

  renderPositions(p.positions || []);
  loadAllocChart(p);
}

function renderPositions(positions) {
  const tbody = $('positions-body');
  if (!tbody) return;
  if (!positions.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="empty-msg">No open positions</td></tr>';
    return;
  }
  const hasExternal = positions.some(p => p.source === 'external');
  tbody.innerHTML = positions.map(p => {
    const srcBadge = p.source === 'external'
      ? `<button class="badge badge-ext adopt-btn" onclick="adoptPosition('${escapeHtml(p.symbol)}')">ext ⟶ adopt</button>`
      : '<span class="badge badge-bot">bot</span>';
    return `
    <tr>
      <td><strong>${escapeHtml(p.symbol)}</strong></td>
      <td style="font-family:var(--mono)">${fmt(p.quantity, 6)}</td>
      <td style="font-family:var(--mono)">${fmtPrice(p.avg_entry_price)}</td>
      <td style="font-family:var(--mono)">${fmtPrice(p.current_price)}</td>
      <td style="font-family:var(--mono)">$${fmt(p.value_usdt)}</td>
      <td class="${pnlClass(p.pnl_usdt)}" style="font-family:var(--mono);font-weight:600">${pnlSign(p.pnl_usdt)}</td>
      <td class="${pnlClass(p.pnl_pct)}" style="font-family:var(--mono);font-weight:600">${pnlSign(p.pnl_pct)}%</td>
      <td style="font-family:var(--mono);font-size:11px">${fmtPrice(p.stop_loss_price)}</td>
      <td style="font-family:var(--mono);font-size:11px">${fmtPrice(p.take_profit_price)}</td>
      <td style="font-size:11px;color:var(--muted)">${holdDuration(p.opened_at)}</td>
      <td>${srcBadge}</td>
    </tr>`;
  }).join('');

  // Show "Adopt All" button when there are external positions
  let adoptAllBtn = document.getElementById('adopt-all-btn');
  if (hasExternal) {
    if (!adoptAllBtn) {
      adoptAllBtn = document.createElement('button');
      adoptAllBtn.id = 'adopt-all-btn';
      adoptAllBtn.className = 'btn btn-adopt-all';
      adoptAllBtn.textContent = 'Adopt All External';
      adoptAllBtn.onclick = () => adoptPosition(null);
      tbody.closest('table')?.parentElement?.insertBefore(adoptAllBtn, tbody.closest('table'));
    }
  } else if (adoptAllBtn) {
    adoptAllBtn.remove();
  }
}

/* ═══════════════════════════════════════════════════════════
   Claude Decision
   ═══════════════════════════════════════════════════════════ */
function renderDecision(d) {
  const el = $('decision-content');
  if (!el) return;

  const actionClass = {BUY:'action-buy', SELL:'action-sell', HOLD:'action-hold'}[d.action] || 'action-hold';
  const confPct = Math.round((d.confidence || 0) * 100);
  const signals = (d.primary_signals || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');
  const risks   = (d.risk_factors || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');

  el.innerHTML = `
    <div class="decision-box">
      <div class="decision-top">
        <div class="decision-action-badge ${actionClass}">${escapeHtml(d.action)}</div>
        <div class="decision-meta-row">
          <span class="dm-symbol">${escapeHtml(d.symbol)}</span>
          <span class="dm-detail">${escapeHtml(d.timeframe)} · ${fmt(d.quantity_pct,1)}% of portfolio</span>
          <div class="confidence-wrap">
            <span class="confidence-label">Conf</span>
            <div class="confidence-bg"><div class="confidence-bar" style="width:${confPct}%"></div></div>
            <span class="confidence-pct">${confPct}%</span>
          </div>
        </div>
      </div>
      <div class="decision-body">
        <div class="decision-section">
          <h3>Primary Signals</h3>
          <ul class="signal-list">${signals || '<li>—</li>'}</ul>
        </div>
        <div class="decision-section">
          <h3>Risk Factors</h3>
          <ul class="signal-list risk-list">${risks || '<li>—</li>'}</ul>
        </div>
        ${d.reasoning ? `<p class="reasoning-text">${escapeHtml(d.reasoning)}</p>` : ''}
      </div>
    </div>`;
}

/* ═══════════════════════════════════════════════════════════
   Bot Status
   ═══════════════════════════════════════════════════════════ */
function renderBotStatus(s) {
  const statusEl = $('stat-status');
  if (statusEl) {
    const dotClass = s.circuit_breaker_tripped ? 'error' : (s.running ? 'running' : 'stopped');
    let label = s.running ? 'RUNNING' : 'STOPPED';
    if (s.circuit_breaker_tripped) label = 'BREAKER';
    statusEl.innerHTML = `<span class="status-dot ${dotClass}"></span> ${label}`;
    statusEl.style.color = s.circuit_breaker_tripped ? 'var(--red)' : (s.running ? 'var(--green)' : 'var(--muted)');
  }

  const startBtn = $('btn-start');
  const stopBtn = $('btn-stop');
  if (startBtn) startBtn.disabled = s.running;
  if (stopBtn) stopBtn.disabled = !s.running;

  // Cycle info
  const cycleInfo = $('stat-cycle-info');
  const cc = s.cycle_count || 0;
  let cycleTxt = `Cycles: ${cc}`;
  if (s.last_cycle_at) {
    const ago = Math.round((Date.now() - new Date(s.last_cycle_at).getTime()) / 1000);
    cycleTxt += ago < 120 ? ` · ${ago}s ago` : ` · ${Math.round(ago / 60)}m ago`;
  }
  if (cycleInfo) cycleInfo.textContent = cycleTxt;

  // Cycle countdown bar
  const barWrap = $('next-cycle-bar');
  const barFill = $('next-cycle-fill');
  const footerNext = $('next-cycle');
  if (s.next_cycle_in_seconds != null && s.running) {
    let pct = 0;
    if (s.last_cycle_at) {
      const elapsed = (Date.now() - new Date(s.last_cycle_at).getTime()) / 1000;
      const total = elapsed + s.next_cycle_in_seconds;
      pct = total > 0 ? Math.max(0, Math.min(100, (elapsed / total) * 100)) : 0;
    }
    if (barWrap) barWrap.style.display = '';
    if (barFill) barFill.style.width = pct + '%';
    const secs = s.next_cycle_in_seconds;
    const countdownTxt = secs >= 60 ? `${Math.floor(secs/60)}m ${secs%60}s` : `${secs}s`;
    if (cycleInfo) cycleInfo.textContent = cycleTxt + ` · next ${countdownTxt}`;
    if (footerNext) footerNext.textContent = `Next cycle: ${countdownTxt}`;
  } else {
    if (barWrap) barWrap.style.display = 'none';
    if (footerNext) footerNext.textContent = '';
  }

  // Mode buttons
  const mode = (s.mode || 'demo').toLowerCase();
  const demoBtn = $('btn-demo');
  const realBtn = $('btn-real');
  if (demoBtn) demoBtn.classList.toggle('active', mode === 'demo');
  if (realBtn) realBtn.classList.toggle('active', mode === 'real');

  // Risk profile
  if (s.risk_profile) {
    syncRiskButtons(s.risk_profile.key || 'balanced');
    const rpEl = $('stat-risk-profile');
    if (rpEl) rpEl.textContent = s.risk_profile.label || s.risk_profile.key || '—';
  }

  // Market regime
  if (s.market_regime && s.market_regime.regime) {
    const regimeEl = $('stat-regime');
    if (regimeEl) {
      const r = s.market_regime;
      const emoji = {strong_uptrend:'🟢',uptrend:'🟡',ranging:'⚪',downtrend:'🟠',strong_downtrend:'🔴',choppy:'⚡'}[r.regime] || '❓';
      regimeEl.textContent = `${emoji} ${r.regime.replace(/_/g,' ').toUpperCase()}`;
      regimeEl.title = r.description || '';
    }
  }

  // Less fear sync
  if (s.less_fear !== undefined) {
    syncLessFearButton(s.less_fear);
  }

  // Lock risk profile sync
  if (s.lock_risk_profile !== undefined) {
    syncLockProfileButton(s.lock_risk_profile);
  }

  // Background processes panel
  if (s.bg_processes) {
    renderBgProcesses(s.bg_processes, s.running);
  }
}

/* ═══════════════════════════════════════════════════════════
   Background Processes
   ═══════════════════════════════════════════════════════════ */
function renderBgProcesses(bg, botRunning) {
  const set = (id, html) => { const el = $(id); if (el) el.innerHTML = html; };
  const setText = (id, txt, cls='') => {
    const el = $(id);
    if (!el) return;
    el.textContent = txt;
    el.className = 'bgproc-value' + (cls ? ' ' + cls : '');
  };

  // Bot loop
  setText('bgproc-bot-loop',
    botRunning ? '● Running' : '○ Stopped',
    botRunning ? 'ok' : 'muted'
  );

  // Startup backtest
  setText('bgproc-backtest',
    bg.startup_backtest_done ? '✓ Done' : '⟳ Running…',
    bg.startup_backtest_done ? 'ok' : 'warn'
  );

  // Last express fired
  const lastExp = $('bgproc-last-express');
  if (lastExp) {
    lastExp.textContent = bg.express_last || '—';
    lastExp.className = 'bgproc-value' + (bg.express_last ? ' warn' : ' muted');
  }

  // Active express tasks
  const expActive = bg.express_active || [];
  setText('bgproc-express',
    expActive.length === 0
      ? 'Idle'
      : `${expActive.length} active: ${expActive.map(s => s.split('/')[0]).join(', ')}`,
    expActive.length > 0 ? 'warn' : 'muted'
  );

  // Scanner hit count
  setText('bgproc-scanner',
    `${bg.scanner_hot_count || 0} hot symbols`,
    bg.scanner_hot_count > 0 ? 'ok' : 'muted'
  );

  // Hot candidate pills (compact row inside bgproc)
  const pillsEl = $('bgproc-hot');
  if (pillsEl) {
    const top = bg.scanner_top || [];
    pillsEl.innerHTML = top.length === 0
      ? '<span class="bgproc-value muted">—</span>'
      : top.map(c => `<span class="bgproc-pill" title="score ${c.score}">${c.symbol.split('/')[0]} ${c.score}</span>`).join('');
  }

  // ── Force-buy table (top 5 hot candidates) ────────────────────────────────
  renderForceBuyTable(bg.scanner_top || [], bg.express_active || []);

  // asyncio task count badge
  const tasks = (bg.asyncio_tasks || []).filter(n => n && n !== 'Task-1');
  const badgeEl = $('bgproc-task-count');
  if (badgeEl) badgeEl.textContent = `${tasks.length} task${tasks.length !== 1 ? 's' : ''}`;

  // asyncio task list
  const taskListEl = $('bgproc-tasks');
  if (taskListEl) {
    if (tasks.length === 0) {
      taskListEl.innerHTML = '<div class="bgproc-task muted">—</div>';
    } else {
      taskListEl.innerHTML = tasks.map(name => {
        let cls = '';
        if (name.startsWith('express_')) cls = ' express';
        else if (name === 'bot_loop') cls = ' bot';
        else if (name.includes('backtest')) cls = ' backtest';
        return `<div class="bgproc-task${cls}" title="${escapeHtml(name)}">${escapeHtml(name)}</div>`;
      }).join('');
    }
  }
}

/* ═══════════════════════════════════════════════════════════
   Force-Buy Panel
   ═══════════════════════════════════════════════════════════ */

function renderForceBuyTable(candidates, expressActive) {
  const tbody = $('force-buy-body');
  const badge = $('force-buy-count');
  if (!tbody) return;

  if (badge) badge.textContent = candidates.length;

  if (candidates.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-msg">No hot candidates yet — scanner runs every 60s</td></tr>';
    return;
  }

  tbody.innerHTML = candidates.map(c => {
    const base = c.symbol.split('/')[0];
    const isRunning = expressActive.includes(c.symbol);
    const pct = c.pct_24h != null ? c.pct_24h : 0;
    const pctCls = pct >= 0 ? 'pos' : 'neg';
    const pctStr = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
    const vol = c.volume_ratio != null ? c.volume_ratio.toFixed(1) + 'x' : '—';
    const reasons = (c.reasons || []).map(r => `<span class="force-reason">${escapeHtml(r)}</span>`).join('');

    let statusHtml, btnHtml;
    if (c.held) {
      statusHtml = '<span class="fb-status fb-held">HELD</span>';
      btnHtml = '<button class="btn-force-buy" disabled>Already held</button>';
    } else if (isRunning) {
      statusHtml = '<span class="fb-status fb-running">RUNNING</span>';
      btnHtml = '<button class="btn-force-buy btn-force-running" disabled>⟳ Analysing…</button>';
    } else if (c.in_cooldown) {
      statusHtml = '<span class="fb-status fb-cooldown">COOLDOWN</span>';
      btnHtml = `<button class="btn-force-buy btn-force-override" onclick="forceBuy('${escapeHtml(c.symbol)}', this)">⚡ Force Override</button>`;
    } else {
      statusHtml = '<span class="fb-status fb-ready">READY</span>';
      btnHtml = `<button class="btn-force-buy btn-force-active" onclick="forceBuy('${escapeHtml(c.symbol)}', this)">⚡ Force Buy</button>`;
    }

    const scoreCls = c.score >= 80 ? 'score-high' : c.score >= 65 ? 'score-mid' : 'score-low';

    return `<tr>
      <td class="fb-symbol"><span class="sym-badge">${escapeHtml(base)}</span></td>
      <td><span class="fb-score ${scoreCls}">${c.score}</span></td>
      <td class="${pctCls}">${pctStr}</td>
      <td>${vol}</td>
      <td class="fb-reasons">${reasons || '<span class="muted-text">—</span>'}</td>
      <td>${statusHtml}</td>
      <td>${btnHtml}</td>
    </tr>`;
  }).join('');
}

async function forceBuy(symbol, btn) {
  if (!confirm(`Force-buy ${symbol}?\n\nThis will immediately trigger the express-lane analysis.\nAll normal risk guards (position size, max positions, cash) still apply.`)) return;

  btn.disabled = true;
  btn.textContent = '⟳ Triggering…';

  try {
    const resp = await fetch(`${API_BASE}/api/bot/force-buy`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({symbol}),
    });
    const data = await resp.json();
    if (!resp.ok) {
      alert(`Force-buy failed: ${data.detail || resp.statusText}`);
      btn.disabled = false;
      btn.textContent = '⚡ Force Buy';
      return;
    }
    if (data.status === 'already_running') {
      btn.textContent = '⟳ Analysing…';
    } else {
      btn.textContent = '✓ Triggered';
      btn.classList.add('btn-force-done');
    }
  } catch (e) {
    alert('Error: ' + e.message);
    btn.disabled = false;
    btn.textContent = '⚡ Force Buy';
  }
}

/* ═══════════════════════════════════════════════════════════
   Risk Profile
   ═══════════════════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════════════════
   Less Fear Toggle
   ═══════════════════════════════════════════════════════════ */
let lessFearActive = false;

function syncLessFearButton(enabled) {
  lessFearActive = enabled;
  const btn = $('btn-less-fear');
  if (!btn) return;
  if (enabled) {
    btn.classList.add('active');
    btn.textContent = '🔥 Less Fear ON';
  } else {
    btn.classList.remove('active');
    btn.textContent = '😱 Less Fear';
  }
}

async function toggleLessFear() {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/less-fear`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({enabled: !lessFearActive}),
    });
    if (!resp.ok) { const e = await resp.json(); alert(e.detail || 'Failed'); return; }
    const data = await resp.json();
    syncLessFearButton(data.enabled);
  } catch (e) { alert('Error: ' + e.message); }
}

async function loadLessFear() {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/less-fear`);
    if (resp.ok) {
      const data = await resp.json();
      syncLessFearButton(data.enabled);
    }
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════════
   Lock Risk Profile Toggle
   ═══════════════════════════════════════════════════════════════ */
let lockProfileActive = false;

function syncLockProfileButton(enabled) {
  lockProfileActive = enabled;
  const btn = $('btn-lock-profile');
  if (!btn) return;
  if (enabled) {
    btn.classList.add('active');
    btn.textContent = '🔒 Profile Locked';
  } else {
    btn.classList.remove('active');
    btn.textContent = '🔒 Lock Profile';
  }
}

async function toggleLockRiskProfile() {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/lock-risk-profile`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({enabled: !lockProfileActive}),
    });
    if (!resp.ok) { const e = await resp.json(); alert(e.detail || 'Failed'); return; }
    const data = await resp.json();
    syncLockProfileButton(data.enabled);
  } catch (e) { alert('Error: ' + e.message); }
}

async function loadLockRiskProfile() {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/lock-risk-profile`);
    if (resp.ok) {
      const data = await resp.json();
      syncLockProfileButton(data.enabled);
    }
  } catch (_) {}
}

/* ─── Adopt external positions ─── */
async function adoptPosition(symbol) {
  const label = symbol || 'all external positions';
  if (!confirm(`Adopt ${label}? Bot will manage SL/TP and exits.`)) return;
  try {
    const body = symbol ? {symbol} : {};
    const resp = await fetch(`${API_BASE}/api/bot/adopt-positions`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    if (!resp.ok) { const e = await resp.json(); alert(e.detail || 'Failed'); return; }
    const data = await resp.json();
    alert(`Adopted ${data.count} position(s): ${data.adopted.join(', ') || 'none'}`);
  } catch (e) { alert('Error: ' + e.message); }
}

/* ═══════════════════════════════════════════════════════════
   Price tick / Market update
   ═══════════════════════════════════════════════════════════ */
function handlePriceTick(data) {
  loadPortfolio();
}

function addChartSymbols(symbols) {
  const sel = $('chart-symbol');
  if (!sel) return;
  symbols.forEach(sym => {
    if (!knownSymbols.has(sym)) {
      knownSymbols.add(sym);
      const opt = document.createElement('option');
      opt.value = sym;
      opt.textContent = sym;
      sel.appendChild(opt);
    }
  });
}

function handleMarketUpdate(data) {
  addChartSymbols(Object.keys(data));
}

async function loadChartSymbols() {
  try {
    const resp = await fetch(`${API_BASE}/api/symbols`);
    const symbols = await resp.json();
    addChartSymbols(symbols);
    const sel = $('chart-symbol');
    if (sel && sel.value === '' && symbols.length > 0) {
      sel.value = symbols[0];
      loadChart();
    }
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   Trade History
   ═══════════════════════════════════════════════════════════ */
async function loadTrades(delta) {
  tradePage = Math.max(1, tradePage + delta);
  try {
    const resp = await fetch(`${API_BASE}/api/trades?page=${tradePage}&page_size=20`);
    const data = await resp.json();
    const pi = $('page-info');
    if (pi) pi.textContent = `Page ${tradePage}`;
    const prevBtn = $('btn-prev');
    if (prevBtn) prevBtn.disabled = tradePage <= 1;

    const tbody = $('trades-body');
    if (!tbody) return;
    const items = data.items || [];
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">No trades yet</td></tr>';
      return;
    }
    tbody.innerHTML = items.map(t => `
      <tr>
        <td style="font-size:11px">${fmtTime(t.created_at)}</td>
        <td><strong>${escapeHtml(t.symbol)}</strong></td>
        <td class="${t.direction==='BUY'?'dir-buy':'dir-sell'}">${t.direction}</td>
        <td><span class="badge badge-${t.mode}">${t.mode.toUpperCase()}</span></td>
        <td style="font-family:var(--mono)">${fmt(t.quantity, 6)}</td>
        <td style="font-family:var(--mono)">${fmtPrice(t.price)}</td>
        <td class="${pnlClass(t.pnl_usdt)}" style="font-family:var(--mono);font-weight:600">${pnlSign(t.pnl_usdt)}</td>
        <td class="${pnlClass(t.pnl_pct)}" style="font-family:var(--mono);font-weight:600">${pnlSign(t.pnl_pct)}%</td>
      </tr>`).join('');
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   Charts
   ═══════════════════════════════════════════════════════════ */
function selectTf(btn, tf) {
  document.querySelectorAll('.btn-tf').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentTf = tf;
  loadChart();
}

async function loadChart() {
  const sym = $('chart-symbol')?.value;
  if (!sym) return;
  try {
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
  } catch (_) {}
}

const chartTheme = {
  grid: '#141628',
  tick: '#6b7094',
  font: 10,
};

function renderPriceChart(labels, closes) {
  if (priceChart) priceChart.destroy();
  const canvas = $('price-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, 'rgba(124,77,255,0.15)');
  gradient.addColorStop(1, 'rgba(124,77,255,0.0)');

  priceChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Price',
        data: closes,
        borderColor: '#7c4dff',
        backgroundColor: gradient,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.2,
        fill: true,
      }]
    },
    options: chartOptions('Price'),
  });
}

function renderRsiChart(labels, rsi) {
  if (rsiChart) rsiChart.destroy();
  const canvas = $('rsi-chart');
  if (!canvas) return;
  rsiChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'RSI 14', data: rsi, borderColor: '#ffd740', borderWidth: 1.5, pointRadius: 0, fill: false },
      ]
    },
    options: {
      ...chartOptions('RSI'),
      plugins: { legend: { display: true, labels: { color: chartTheme.tick, font: { size: 9 } } } },
    },
  });
}

function renderMacdChart(labels, macd) {
  if (macdChart) macdChart.destroy();
  const canvas = $('macd-chart');
  if (!canvas) return;
  macdChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'MACD Hist',
        data: macd.hist,
        backgroundColor: macd.hist.map(v => v >= 0 ? 'rgba(0,230,118,0.5)' : 'rgba(255,82,82,0.5)'),
        borderRadius: 1,
      }]
    },
    options: chartOptions('MACD'),
  });
}

function chartOptions(label) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { display: false }, tooltip: { mode: 'index' } },
    scales: {
      x: { ticks: { color: chartTheme.tick, maxTicksLimit: 8, font: { size: chartTheme.font } }, grid: { color: chartTheme.grid } },
      y: { ticks: { color: chartTheme.tick, font: { size: chartTheme.font } }, grid: { color: chartTheme.grid } },
    },
  };
}

/* ─── RSI calc ─── */
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

/* ─── MACD calc ─── */
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

/* ═══════════════════════════════════════════════════════════
   Portfolio Growth Chart
   ═══════════════════════════════════════════════════════════ */
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
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 220);
    gradient.addColorStop(0, 'rgba(124,77,255,0.30)');
    gradient.addColorStop(1, 'rgba(124,77,255,0.01)');

    if (growthChart) growthChart.destroy();
    growthChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Portfolio Value',
          data: values,
          borderColor: '#7c4dff',
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
          x: { ticks: { color: chartTheme.tick, maxTicksLimit: 8, font: { size: chartTheme.font } }, grid: { color: chartTheme.grid } },
          y: { ticks: { color: chartTheme.tick, font: { size: chartTheme.font }, callback: v => `$${fmt(v, 0)}` }, grid: { color: chartTheme.grid } },
        },
      }
    });
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   Allocation Donut
   ═══════════════════════════════════════════════════════════ */
function loadAllocChart(portfolio) {
  if (!portfolio) return;

  const labels = ['Cash'];
  const values = [Math.max(portfolio.cash_usdt || 0, 0)];
  const colors = ['#3a3f5c'];

  (portfolio.positions || []).forEach((p, i) => {
    labels.push(p.symbol.replace(/\/USD[TC]/, ''));
    values.push(Math.max(p.value_usdt || 0, 0));
    colors.push(PALETTE[i % PALETTE.length]);
  });

  const total = values.reduce((a, b) => a + b, 0) || 1;

  if (allocChart) allocChart.destroy();
  const canvas = $('alloc-chart');
  if (!canvas) return;
  allocChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: '#12141e',
        borderWidth: 3,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
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
      `<div class="alloc-legend-item"><div class="alloc-dot" style="background:${colors[i]}"></div>${escapeHtml(l)} ${(values[i] / total * 100).toFixed(1)}%</div>`
    ).join('');
  }
}

/* ═══════════════════════════════════════════════════════════
   Trade P&L Chart
   ═══════════════════════════════════════════════════════════ */
async function loadPnlChart() {
  try {
    const resp = await fetch(`${API_BASE}/api/trades?page=1&page_size=100`);
    const data = await resp.json();
    const sells = (data.items || [])
      .filter(t => t.direction === 'SELL' && t.pnl_usdt != null)
      .reverse();

    const canvas = $('pnl-chart');
    if (!canvas) return;

    if (!sells.length) {
      if (pnlChart) { pnlChart.destroy(); pnlChart = null; }
      return;
    }

    const labels = sells.map(t => t.symbol.replace(/\/USD[TC]/, ''));
    const values = sells.map(t => t.pnl_usdt);

    if (pnlChart) pnlChart.destroy();
    pnlChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'P&L ($)',
          data: values,
          backgroundColor: values.map(v => v >= 0 ? 'rgba(0,230,118,0.6)' : 'rgba(255,82,82,0.6)'),
          borderColor: values.map(v => v >= 0 ? '#00e676' : '#ff5252'),
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
          x: { ticks: { color: chartTheme.tick, font: { size: chartTheme.font }, maxRotation: 45 }, grid: { display: false } },
          y: { ticks: { color: chartTheme.tick, font: { size: chartTheme.font }, callback: v => `$${fmt(v, 0)}` }, grid: { color: chartTheme.grid } },
        }
      }
    });
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   Analytics Stats
   ═══════════════════════════════════════════════════════════ */
async function loadAnalytics() {
  try {
    const resp = await fetch(`${API_BASE}/api/analytics`);
    if (!resp.ok) return;
    const d = await resp.json();

    // Analytics stats in PnL card header
    const el = $('analytics-stats');
    if (el) {
      const wr    = d.closed_trades > 0 ? `${Math.round(d.win_rate * 100)}%` : '—';
      const avgPnl = d.closed_trades > 0 ? `${d.avg_pnl_usdt >= 0 ? '+' : ''}$${fmt(d.avg_pnl_usdt)}` : '—';
      const best  = d.best_pnl_usdt  > 0 ? `+$${fmt(d.best_pnl_usdt)}`              : '—';
      const worst = d.worst_pnl_usdt < 0 ? `-$${fmt(Math.abs(d.worst_pnl_usdt))}` : '—';
      const wrColor = d.win_rate >= 0.5 ? 'var(--green)' : 'var(--red)';

      el.innerHTML = `
        <div class="a-stat"><div class="a-val">${d.closed_trades}</div><div class="a-lbl">Closed</div></div>
        <div class="a-stat"><div class="a-val" style="color:${wrColor}">${wr}</div><div class="a-lbl">Win Rate</div></div>
        <div class="a-stat"><div class="a-val ${d.avg_pnl_usdt >= 0 ? 'pnl-pos' : 'pnl-neg'}">${avgPnl}</div><div class="a-lbl">Avg P&L</div></div>
        <div class="a-stat"><div class="a-val pnl-pos">${best}</div><div class="a-lbl">Best</div></div>
        <div class="a-stat"><div class="a-val pnl-neg">${worst}</div><div class="a-lbl">Worst</div></div>`;
    }

    // Metrics strip values
    const ddEl = $('stat-drawdown');
    if (ddEl) {
      ddEl.textContent = d.max_drawdown_pct != null ? `${fmt(d.max_drawdown_pct, 1)}%` : '—';
      ddEl.style.color = d.max_drawdown_pct > 10 ? 'var(--red)' : d.max_drawdown_pct > 5 ? 'var(--yellow)' : 'var(--green)';
    }
    const shEl = $('stat-sharpe');
    if (shEl) {
      shEl.textContent = d.sharpe_ratio != null ? fmt(d.sharpe_ratio, 2) : '—';
      if (d.sharpe_ratio != null) shEl.style.color = d.sharpe_ratio >= 1 ? 'var(--green)' : d.sharpe_ratio >= 0 ? 'var(--muted)' : 'var(--red)';
    }
    const wrEl = $('stat-win-rate');
    if (wrEl && d.closed_trades > 0) {
      const wrPct = Math.round(d.win_rate * 100);
      wrEl.textContent = `${wrPct}%`;
      wrEl.style.color = wrPct >= 50 ? 'var(--green)' : 'var(--red)';
    }
    const stEl = $('stat-streak');
    if (stEl) {
      const cw = d.max_consecutive_wins || 0;
      const cl = d.max_consecutive_losses || 0;
      stEl.innerHTML = `<span style="color:var(--green)">${cw}W</span>/<span style="color:var(--red)">${cl}L</span>`;
    }
    const feEl = $('stat-fees');
    if (feEl) feEl.textContent = d.total_fees_usdt ? `$${fmt(d.total_fees_usdt)}` : '$0.00';

    // Bot Realized P&L
    const botPnlEl = $('stat-bot-pnl');
    if (botPnlEl && d.total_realized_pnl != null) {
      const rp = d.total_realized_pnl;
      botPnlEl.textContent = `${rp >= 0 ? '+' : ''}$${fmt(rp)}`;
      botPnlEl.className = 'kpi-value ' + (rp >= 0 ? 'pnl-pos' : 'pnl-neg');
    }
    const botRecEl = $('stat-bot-record');
    if (botRecEl && d.closed_trades > 0) {
      botRecEl.textContent = `${d.wins}W ${d.losses}L (${Math.round(d.win_rate * 100)}%)`;
    }
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   AI Decision Feed
   ═══════════════════════════════════════════════════════════ */
async function loadDecisionFeed() {
  try {
    const resp = await fetch(`${API_BASE}/api/decisions?page=1&page_size=25`);
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
      const signal = (d.primary_signals && d.primary_signals[0]) || (d.reasoning || '').slice(0, 60) || '—';
      const sym = d.symbol || 'MARKET';
      return `
        <div class="feed-item">
          <span class="feed-badge ${badgeCls}">${d.action}</span>
          <span class="feed-sym">${escapeHtml(sym)}</span>
          <span class="feed-snippet" title="${escapeHtml((d.primary_signals || []).join(' | '))}">${escapeHtml(signal)}</span>
          <div style="display:flex;align-items:center;gap:4px;min-width:70px;">
            <div class="feed-conf-bg"><div class="feed-conf-bar" style="width:${confPct}%"></div></div>
            <span style="font-size:10px;color:var(--muted);font-family:var(--mono)">${confPct}%</span>
          </div>
          <span class="feed-time">${timeAgo(d.created_at)}</span>
        </div>`;
    }).join('');
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   GPU Models Status
   ═══════════════════════════════════════════════════════════ */
const GPU_MODELS = [
  { key: 'transformer_trained', name: 'Transformer', icon: '🧠' },
  { key: 'lstm_trained',        name: 'LSTM',        icon: '📈' },
  { key: 'rl_trained',          name: 'Dueling DQN', icon: '🎮' },
  { key: 'sentiment_loaded',    name: 'Sentiment',   icon: '💬' },
  { key: 'mtf_trained',         name: 'MTF Fusion',  icon: '🔀' },
  { key: 'volatility_trained',  name: 'Volatility',  icon: '📊' },
  { key: 'anomaly_trained',     name: 'Anomaly',     icon: '⚠' },
  { key: 'exit_trained',        name: 'Exit RL',     icon: '🚪' },
  { key: 'attention_ready',     name: 'Attention',   icon: '👁' },
  { key: 'correlation_ready',   name: 'Correlation', icon: '🔗' },
];

async function loadGpuStatus() {
  const statusEl = $('stat-gpu-status');
  const infoEl = $('stat-gpu-info');
  const modelsGrid = $('gpu-models-grid');
  const modelsCount = $('gpu-models-count');

  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    if (!resp.ok) return;
    const data = await resp.json();
    const gpu = data.gpu_server;

    if (!gpu) {
      if (statusEl) statusEl.innerHTML = '<span class="status-dot stopped"></span> OFF';
      if (infoEl) infoEl.textContent = 'Not configured';
      if (modelsGrid) modelsGrid.innerHTML = '<div class="empty-msg">No GPU server configured</div>';
      if (modelsCount) modelsCount.textContent = '0/10';
      return;
    }

    if (gpu.status === 'unreachable') {
      if (statusEl) statusEl.innerHTML = '<span class="status-dot error"></span> DOWN';
      if (infoEl) infoEl.textContent = 'Unreachable';
      if (modelsGrid) modelsGrid.innerHTML = '<div class="empty-msg">GPU server unreachable</div>';
      if (modelsCount) modelsCount.textContent = '0/10';
      return;
    }

    // GPU server active
    if (statusEl) statusEl.innerHTML = '<span class="status-dot running"></span> ON';
    const gpuName = (gpu.gpu || 'CPU').replace(/NVIDIA\s*/i, '');
    const vram = gpu.vram_gb ? ` ${gpu.vram_gb}GB` : '';
    if (infoEl) infoEl.textContent = `${gpuName}${vram}`;

    // Render model status grid
    if (modelsGrid) {
      let readyCount = 0;
      modelsGrid.innerHTML = GPU_MODELS.map(m => {
        const isReady = gpu[m.key];
        if (isReady) readyCount++;
        const dotCls = isReady ? 'ready' : 'offline';
        const tagCls = isReady ? 'gpu-tag-ready' : 'gpu-tag-off';
        const tagTxt = isReady ? 'READY' : 'OFF';
        return `
          <div class="gpu-model-item">
            <span class="gpu-model-dot ${dotCls}"></span>
            <span class="gpu-model-name">${m.icon} ${m.name}</span>
            <span class="gpu-model-tag ${tagCls}">${tagTxt}</span>
          </div>`;
      }).join('');
      if (modelsCount) modelsCount.textContent = `${readyCount}/${GPU_MODELS.length}`;
    }
  } catch (_) {
    if (modelsGrid) modelsGrid.innerHTML = '<div class="empty-msg">Failed to check GPU</div>';
  }
}

/* ═══════════════════════════════════════════════════════════
   Claude Cost & Usage
   ═══════════════════════════════════════════════════════════ */
async function loadCreditBalance() {
  try {
    const resp = await fetch(`${API_BASE}/api/usage`);
    if (!resp.ok) return;
    const data = await resp.json();

    const el = $('stat-credits');
    if (!el) return;
    const cost = Number(data.total_cost_usd || 0);
    const calls = data.total_calls || 0;
    const skipped = data.calls_skipped || 0;
    const sonnet = data.calls_sonnet || 0;
    const haiku = data.calls_haiku || 0;
    el.textContent = `$${cost.toFixed(4)}`;
    el.title =
      `Calls: ${calls} (${sonnet}S/${haiku}H/${skipped}skip) | ` +
      `Input: ${((data.total_input_tokens || 0)/1000).toFixed(1)}k | ` +
      `Output: ${((data.total_output_tokens || 0)/1000).toFixed(1)}k | ` +
      `Cache: ${((data.total_cache_read_tokens || 0)/1000).toFixed(1)}k hit`;

    // Model mix
    const mmEl = $('stat-model-mix');
    if (mmEl && calls > 0) {
      mmEl.innerHTML = `<span style="color:var(--green)">${haiku}H</span>/<span style="color:var(--blue)">${sonnet}S</span>`;
      mmEl.title = `Haiku (cheap): ${haiku} | Sonnet (powerful): ${sonnet}`;
    }

    // Cache hit rate
    const chEl = $('stat-cache-hit');
    if (chEl) {
      const totalIn = data.total_input_tokens || 0;
      const cacheRead = data.total_cache_read_tokens || 0;
      const hitPct = totalIn > 0 ? Math.round(cacheRead / totalIn * 100) : 0;
      chEl.textContent = totalIn > 0 ? `${hitPct}%` : '—';
      chEl.style.color = hitPct > 50 ? 'var(--green)' : hitPct > 20 ? 'var(--muted)' : 'var(--red)';
    }

    // Skipped cycles
    const skEl = $('stat-skipped');
    if (skEl) {
      const totalCycles = calls + skipped;
      const skipPct = totalCycles > 0 ? Math.round(skipped / totalCycles * 100) : 0;
      skEl.textContent = totalCycles > 0 ? `${skipped} (${skipPct}%)` : '—';
    }
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   API Calls
   ═══════════════════════════════════════════════════════════ */
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
    const demoBtn = $('btn-demo');
    const realBtn = $('btn-real');
    if (demoBtn) demoBtn.classList.toggle('active', mode === 'demo');
    if (realBtn) realBtn.classList.toggle('active', mode === 'real');
  } catch (e) { alert('Error: ' + e.message); }
}

async function startBot() {
  await fetch(`${API_BASE}/api/bot/start`, {method:'POST'});
}

async function stopBot() {
  await fetch(`${API_BASE}/api/bot/stop`, {method:'POST'});
}

async function resetDemo() {
  if (!confirm('Reset demo? This will wipe ALL trade history, decisions, and positions and restore the initial balance.')) return;
  const resp = await fetch(`${API_BASE}/api/bot/reset-demo`, {method:'POST'});
  if (resp.ok) {
    await Promise.all([
      loadPortfolio(), loadTrades(0), syncBotStatus(), loadLastDecision(),
      loadPnlChart(), loadAnalytics(), loadDecisionFeed(), loadGrowthChart(),
    ]);
    const dc = $('decision-content');
    if (dc) dc.innerHTML = '<div class="decision-empty">History cleared. Start the bot to begin.</div>';
  } else {
    alert('Reset failed');
  }
}

/* ─── Bot status REST poll ─── */
async function syncBotStatus() {
  try {
    const resp = await fetch(`${API_BASE}/api/bot/status`);
    const data = await resp.json();
    renderBotStatus(data);
  } catch (_) {}
}

/* ─── Last decision REST fetch ─── */
async function loadLastDecision() {
  try {
    const resp = await fetch(`${API_BASE}/api/decisions?page=1&page_size=1`);
    const data = await resp.json();
    if (data.items && data.items.length) renderDecision(data.items[0]);
  } catch (_) {}
}

/* ═══════════════════════════════════════════════════════════
   Binance Account
   ═══════════════════════════════════════════════════════════ */
async function loadExchangeAccount() {
  try {
    const resp = await fetch(`${API_BASE}/api/exchange-account`);
    const data = await resp.json();
    renderExchangeAccount(data);
  } catch (e) {
    const tbody = $('exchange-body');
    if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">Failed to load exchange data</td></tr>';
  }
}

function renderExchangeAccount(data) {
  const summary = $('exchange-summary');
  const tbody = $('exchange-body');
  if (!tbody) return;

  if (data.error) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-msg">${escapeHtml(data.error)}</td></tr>`;
    if (summary) summary.innerHTML = '';
    return;
  }

  if (summary) {
    const stables = data.assets.filter(a => a.type === 'stablecoin');
    const cryptos = data.assets.filter(a => a.type === 'crypto');
    const fiats = data.assets.filter(a => a.type === 'fiat');
    const cashVal = stables.reduce((s, a) => s + a.value_usdt, 0);
    const cryptoVal = cryptos.reduce((s, a) => s + a.value_usdt, 0);
    const fiatVal = fiats.reduce((s, a) => s + a.value_usdt, 0);
    summary.innerHTML = `
      <span class="exch-stat">Total: <strong>$${fmt(data.total_value_usdt)}</strong></span>
      <span class="exch-stat">Cash: <strong>$${fmt(cashVal)}</strong></span>
      <span class="exch-stat">Crypto: <strong>$${fmt(cryptoVal)}</strong></span>
      ${fiatVal > 0 ? `<span class="exch-stat">Fiat: <strong>$${fmt(fiatVal)}</strong></span>` : ''}
      <span class="exch-stat">${data.num_assets} assets</span>`;
  }

  if (!data.assets || !data.assets.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">No assets found</td></tr>';
    return;
  }

  tbody.innerHTML = data.assets.map(a => {
    const typeIcon = {stablecoin: '💵', crypto: '🪙', fiat: '🏦'}[a.type] || '';
    const managed = a.managed_by === 'bot' ? '<span class="badge badge-bot">bot</span>'
                  : a.managed_by === 'external' ? '<span class="badge badge-ext">tracked</span>'
                  : '<span class="badge badge-none">—</span>';
    const valClass = a.value_usdt < 1 ? 'dust-row' : '';
    return `
    <tr class="${valClass}">
      <td><strong>${escapeHtml(a.asset)}</strong></td>
      <td>${typeIcon} ${escapeHtml(a.type)}</td>
      <td style="font-family:var(--mono)">${a.total < 0.001 ? a.total.toExponential(2) : fmt(a.total, 6)}</td>
      <td style="font-family:var(--mono)">${a.free < 0.001 ? a.free.toExponential(2) : fmt(a.free, 6)}</td>
      <td style="font-family:var(--mono)">${a.locked > 0 ? fmt(a.locked, 6) : '—'}</td>
      <td style="font-family:var(--mono)">${a.pair ? fmtPrice(a.price) : '—'}</td>
      <td style="font-family:var(--mono);font-weight:${a.value_usdt >= 1 ? '600' : '400'}">$${fmt(a.value_usdt)}</td>
      <td>${managed}</td>
    </tr>`;
  }).join('');
}

/* ═══════════════════════════════════════════════════════════
   Footer Clock
   ═══════════════════════════════════════════════════════════ */
function updateFooterClock() {
  const el = $('footer-time');
  if (el) {
    el.textContent = new Date().toLocaleString('pl-PL', {
      timeZone: 'Europe/Warsaw',
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }) + ' CET';
  }
}

/* ═══════════════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════════════ */
(async () => {
  connectWs();

  // Restore full UI state from REST on page load
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
    loadExchangeAccount(),
    loadChartSymbols(),
    loadLessFear(),
    loadLockRiskProfile(),
    loadGpuStatus(),
  ]);

  // Footer clock
  updateFooterClock();
  setInterval(updateFooterClock, 1000);

  // Periodic fallback polls
  setInterval(loadPortfolio, 60_000);
  setInterval(syncBotStatus, 30_000);
  setInterval(loadCreditBalance, 300_000);
  setInterval(loadGrowthChart, 60_000);
  setInterval(loadAnalytics, 60_000);
  setInterval(loadDecisionFeed, 30_000);
  setInterval(loadExchangeAccount, 120_000);
  setInterval(loadGpuStatus, 60_000);

  // Orientation change: resize all Chart.js instances so they fill new dimensions
  window.addEventListener('orientationchange', () => {
    setTimeout(() => {
      [priceChart, rsiChart, macdChart, growthChart, allocChart, pnlChart]
        .forEach(c => { if (c) c.resize(); });
    }, 300); // wait for the browser to finish repainting
  });
})();
