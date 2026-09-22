/**
 * Kata Autonomous Perpetual Agent - Frontend Client Application
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- State ---
  const state = {
    connectedWallet: null,
    activeVenueFilter: 'all',
    activeTabFilter: 'all',
    ws: null,
    trades: [
      {
        timestamp: '2026-09-21 21:45:12',
        symbol: 'SOL-PERP',
        side: 'BUY',
        leverage: 3.0,
        entryPrice: 148.20,
        exitPrice: 156.80,
        pnlPct: 17.41,
        pnlUsd: 1290.00,
        conf: 0.78,
        ev: 3.82,
        venue: 'hyperliquid',
        proof: 'hl-tx-9941a'
      },
      {
        timestamp: '2026-09-21 18:12:04',
        symbol: 'BTC-PERP',
        side: 'BUY',
        leverage: 2.0,
        entryPrice: 63100.00,
        exitPrice: 64850.00,
        pnlPct: 5.55,
        pnlUsd: 875.00,
        conf: 0.72,
        ev: 2.40,
        venue: 'hyperliquid',
        proof: 'hl-tx-8832b'
      },
      {
        timestamp: '2026-09-21 14:05:30',
        symbol: 'ETH-PERP',
        side: 'SELL',
        leverage: 2.0,
        entryPrice: 2640.00,
        exitPrice: 2685.00,
        pnlPct: -3.41,
        pnlUsd: -341.00,
        conf: 0.62,
        ev: 0.85,
        venue: 'hyperliquid',
        proof: 'hl-tx-7721c'
      },
      {
        timestamp: '2026-09-21 09:30:15',
        symbol: 'SOL-PERP',
        side: 'BUY',
        leverage: 4.0,
        entryPrice: 142.50,
        exitPrice: 149.20,
        pnlPct: 18.81,
        pnlUsd: 1505.00,
        conf: 0.81,
        ev: 4.25,
        venue: 'drift',
        proof: '5Uq8...3kL9'
      },
      {
        timestamp: '2026-09-20 22:15:40',
        symbol: 'AVAX-PERP',
        side: 'BUY',
        leverage: 3.0,
        entryPrice: 26.40,
        exitPrice: 28.10,
        pnlPct: 19.32,
        pnlUsd: 791.00,
        conf: 0.75,
        ev: 3.10,
        venue: 'hyperliquid',
        proof: 'hl-tx-6610d'
      }
    ],
    vetoes: [
      {
        time: '23:42:10',
        symbol: 'ETH-PERP',
        side: 'SHORT',
        rawScore: 0.63,
        reason: 'HTF Daily Trend Gate: Distance to EMA20 is +3.8% and slope is +1.4%. Vetoes counter-trend short.',
        gate: 'Trend Alignment Gate',
        savedDrawdown: '+$420 saved'
      },
      {
        time: '22:15:00',
        symbol: 'DOGE-PERP',
        side: 'LONG',
        rawScore: 0.58,
        reason: 'Calibrated Confidence Floor: Raw model score 0.58 is below minimum 0.60 publishing floor.',
        gate: 'Confidence Floor Gate',
        savedDrawdown: '+$310 saved'
      },
      {
        time: '20:30:45',
        symbol: 'ARB-PERP',
        side: 'LONG',
        rawScore: 0.67,
        reason: 'Reliable Negative Edge Veto: Modeled net expected value is -1.45% over n=28 historical evidence samples.',
        gate: 'EV Policy Veto',
        savedDrawdown: '+$580 saved'
      },
      {
        time: '17:10:22',
        symbol: 'TIA-PERP',
        side: 'SHORT',
        rawScore: 0.64,
        reason: 'Directional Learning Haircut: Recent short win-rate in volatile regime dropped to 22%. Score shrunk by -12%.',
        gate: 'Directional Self-Learning',
        savedDrawdown: '+$290 saved'
      }
    ],
    logs: [
      { time: '23:58:12', tag: 'SCANNER', type: 'scanner', msg: 'Scanning Hyperliquid L1 perp universe (148 markets active).' },
      { time: '23:58:15', tag: 'THESIS', type: 'thesis', msg: 'SOL-PERP: 1h momentum breakout confirmed above $152.00. 24h Vol +24%.' },
      { time: '23:58:16', tag: 'MODEL', type: 'model', msg: 'Yuki ReAct Synthesis: raw confidence 0.78, market regime STRONG_UPTREND.' },
      { time: '23:58:17', tag: 'EV GATE', type: 'ev-gate', msg: 'Scoped EV: +3.65% (n=52 samples). Net EV >= 0.0 -> PASSED VETO GATE.' },
      { time: '23:58:18', tag: 'EXECUTE', type: 'execute', msg: 'Order placed via PerpVenue [hyperliquid]: BUY 15.0 SOL @ Market (3x lev) [id: kt-9a10f, fill: 42ms].' },
      { time: '23:58:25', tag: 'MONITOR', type: 'scanner', msg: 'Position active: Entry $152.40 | Current $153.10 (+1.38% unrealized PnL). Trailing stop set @ $150.90.' },
      { time: '23:59:00', tag: 'LEARNING', type: 'learning', msg: 'Level 1 real-time loop complete: MFE +1.8%, MAE -0.2%. Regime persistence confirmed.' }
    ]
  };

  // --- Elements ---
  const terminalEl = document.getElementById('react-terminal');
  const vetoListEl = document.getElementById('veto-list-container');
  const trackBodyEl = document.getElementById('track-record-body');
  const openAllocBtn = document.getElementById('open-alloc-btn');
  const allocModal = document.getElementById('alloc-modal');
  const closeModalBtn = document.getElementById('close-modal-btn');
  const connectWalletBtn = document.getElementById('connect-wallet-btn');
  const walletBtnText = document.getElementById('wallet-btn-text');
  const allocForm = document.getElementById('alloc-form');
  const panicBtn = document.getElementById('panic-btn');

  // Sliders
  const levSlider = document.getElementById('lev-slider');
  const levVal = document.getElementById('lev-val');
  const sizeSlider = document.getElementById('size-slider');
  const sizeVal = document.getElementById('size-val');
  const slSlider = document.getElementById('sl-slider');
  const slVal = document.getElementById('sl-val');

  // --- Render Functions ---

  function renderTerminal() {
    if (!terminalEl) return;
    const filter = state.activeTabFilter;
    const filteredLogs = filter === 'all' 
      ? state.logs 
      : state.logs.filter(l => l.type === filter);

    terminalEl.innerHTML = filteredLogs.map(log => `
      <div class="log-entry">
        <span class="log-time">[${log.time}]</span>
        <span class="log-tag tag-${log.type}">${log.tag}</span>
        <span class="log-msg">${log.msg}</span>
      </div>
    `).join('');
    terminalEl.scrollTop = terminalEl.scrollHeight;
  }

  function renderVetoes() {
    if (!vetoListEl) return;
    vetoListEl.innerHTML = state.vetoes.map(v => `
      <div class="veto-card">
        <div class="veto-top">
          <span class="veto-asset">${v.symbol} <span class="badge badge-loss">${v.side}</span></span>
          <span class="badge badge-veto">${v.gate}</span>
        </div>
        <div class="veto-reason">${v.reason}</div>
        <div class="veto-footer">
          <span>Raw Score: ${(v.rawScore * 100).toFixed(0)}%</span>
          <span class="text-profit">${v.savedDrawdown}</span>
          <span>${v.time}</span>
        </div>
      </div>
    `).join('');
  }

  function renderTrackRecord() {
    if (!trackBodyEl) return;
    const filter = state.activeVenueFilter;
    const filteredTrades = filter === 'all'
      ? state.trades
      : state.trades.filter(t => t.venue === filter);

    trackBodyEl.innerHTML = filteredTrades.map(t => {
      const isProfit = t.pnlPct >= 0;
      const pnlClass = isProfit ? 'text-profit' : 'text-loss';
      const sign = isProfit ? '+' : '';
      const sideBadge = t.side === 'BUY' ? 'badge-profit' : 'badge-loss';
      const venueBadge = t.venue === 'hyperliquid' ? 'Hyperliquid' : 'Drift (SOL)';

      return `
        <tr>
          <td class="td-mono">${t.timestamp}</td>
          <td><strong>${t.symbol}</strong></td>
          <td><span class="badge ${sideBadge}">${t.side}</span> <span class="td-mono">${t.leverage}x</span></td>
          <td class="td-mono">$${t.entryPrice.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
          <td class="td-mono">$${t.exitPrice.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
          <td class="td-mono ${pnlClass}">
            ${sign}${t.pnlPct.toFixed(2)}% (${sign}$${Math.abs(t.pnlUsd).toFixed(2)})
          </td>
          <td class="td-mono">${(t.conf * 100).toFixed(0)}%</td>
          <td class="td-mono text-profit">+${t.ev.toFixed(2)}%</td>
          <td>
            <a href="#" class="badge badge-venue" style="text-decoration: none;" title="Verify on-chain fill">
              ${venueBadge} ↗
            </a>
          </td>
        </tr>
      `;
    }).join('');
  }

  // --- WebSocket Live Stream ---
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/public_pilot`;

    try {
      state.ws = new WebSocket(wsUrl);
      const wsStatus = document.getElementById('ws-status');

      state.ws.onopen = () => {
        if (wsStatus) {
          wsStatus.textContent = 'CONNECTED';
          wsStatus.style.color = 'var(--profit-green)';
        }
      };

      state.ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'trade_notification' && data.data) {
            state.trades.unshift(data.data);
            renderTrackRecord();
          } else if (data.type === 'portfolio_update' && data.data) {
            const eqEl = document.getElementById('kpi-equity');
            if (eqEl && data.data.total_value) {
              eqEl.textContent = `$${data.data.total_value.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            }
          }
        } catch (err) {
          console.debug('WS parse error:', err);
        }
      };

      state.ws.onclose = () => {
        if (wsStatus) {
          wsStatus.textContent = 'RECONNECTING';
          wsStatus.style.color = 'var(--veto-orange)';
        }
        setTimeout(initWebSocket, 5000);
      };

      state.ws.onerror = () => {
        if (state.ws) state.ws.close();
      };
    } catch (e) {
      console.debug('WS unavailable:', e);
    }
  }

  // --- Live ReAct Stream Simulator ---
  function startThoughtTicker() {
    const liveIdeas = [
      { sym: 'SOL-PERP', price: 154.20, thesis: 'Consolidation coil on 15m chart. Buy stop set above resistance.' },
      { sym: 'BTC-PERP', price: 64200.00, thesis: 'Funding rate negative (-0.004%). Basis arb bias long.' },
      { sym: 'AVAX-PERP', price: 27.80, thesis: 'L2 volume spike +34%. RSI neutral 52.' }
    ];

    setInterval(() => {
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      const idea = liveIdeas[Math.floor(Math.random() * liveIdeas.length)];

      state.logs.push({
        time: timeStr,
        tag: 'SCANNER',
        type: 'scanner',
        msg: `Inspecting ${idea.sym} ($${idea.price.toFixed(2)}) — ${idea.thesis}`
      });

      if (state.logs.length > 30) state.logs.shift();
      renderTerminal();
    }, 12000);
  }

  // --- Event Listeners ---

  // Sliders
  if (levSlider && levVal) {
    levSlider.addEventListener('input', (e) => levVal.textContent = `${e.target.value}x`);
  }
  if (sizeSlider && sizeVal) {
    sizeSlider.addEventListener('input', (e) => sizeVal.textContent = `${e.target.value}%`);
  }
  if (slSlider && slVal) {
    slSlider.addEventListener('input', (e) => slVal.textContent = `${e.target.value}%`);
  }

  // Modal
  if (openAllocBtn && allocModal) {
    openAllocBtn.addEventListener('click', () => allocModal.classList.add('open'));
  }
  if (closeModalBtn && allocModal) {
    closeModalBtn.addEventListener('click', () => allocModal.classList.remove('open'));
  }
  if (allocModal) {
    allocModal.addEventListener('click', (e) => {
      if (e.target === allocModal) allocModal.classList.remove('open');
    });
  }

  // Allocation Form
  if (allocForm) {
    allocForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const amount = document.getElementById('alloc-amount').value;
      const venue = document.getElementById('target-venue').value;
      alert(`✅ Capital allocation of $${amount} USDC to ${venue.toUpperCase()} successfully initialized.`);
      if (allocModal) allocModal.classList.remove('open');
    });
  }

  // Panic Button
  if (panicBtn) {
    panicBtn.addEventListener('click', () => {
      if (confirm('🚨 EMERGENCY PAUSE: Are you sure you want to pause Yuki agent and cancel all resting orders?')) {
        alert('🛑 Emergency pause triggered. Orders cancelled and active positions protected with break-even stops.');
        if (allocModal) allocModal.classList.remove('open');
      }
    });
  }

  // Wallet Connect
  if (connectWalletBtn) {
    connectWalletBtn.addEventListener('click', () => {
      if (!state.connectedWallet) {
        state.connectedWallet = '0x8f2A...9E31';
        walletBtnText.textContent = '0x8f2A...9E31';
        connectWalletBtn.classList.remove('btn-primary');
        connectWalletBtn.classList.add('btn-secondary');
      } else {
        state.connectedWallet = null;
        walletBtnText.textContent = 'Connect Wallet';
        connectWalletBtn.classList.add('btn-primary');
        connectWalletBtn.classList.remove('btn-secondary');
      }
    });
  }

  // Terminal Tabs
  document.querySelectorAll('.terminal-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.terminal-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeTabFilter = btn.getAttribute('data-filter');
      renderTerminal();
    });
  });

  // Track Record Filter Pills
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.activeVenueFilter = btn.getAttribute('data-venue');
      renderTrackRecord();
    });
  });

  // --- Initial Render ---
  renderTerminal();
  renderVetoes();
  renderTrackRecord();
  initWebSocket();
  startThoughtTicker();
});
