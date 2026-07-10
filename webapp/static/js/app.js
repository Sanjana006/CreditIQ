/* ═══════════════════════════════════════════════════
   CreditIQ — Application Logic & Chart Rendering
   ═══════════════════════════════════════════════════ */

'use strict';

// ─── Chart defaults ────────────────────────────────────────────────────────
Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.font.size   = 12;
Chart.defaults.color       = '#64748b';
Chart.defaults.plugins.legend.display = false;
Chart.defaults.plugins.tooltip.backgroundColor = '#1e293b';
Chart.defaults.plugins.tooltip.titleFont = { weight: '600', size: 12 };
Chart.defaults.plugins.tooltip.bodyFont  = { size: 12 };
Chart.defaults.plugins.tooltip.padding   = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 6;

const PALETTE = {
  blue:   '#1d4ed8',
  blue2:  '#3b82f6',
  blue3:  '#93c5fd',
  green:  '#15803d',
  amber:  '#d97706',
  red:    '#dc2626',
  slate:  '#475569',
  bands: {
    'Very Low':  '#15803d',
    'Low':       '#16a34a',
    'Moderate':  '#d97706',
    'High':      '#ea580c',
    'Very High': '#dc2626',
  }
};

// ─── Utility ───────────────────────────────────────────────────────────────
function fmtPct(v) { return v.toFixed(1) + '%'; }
function bandColor(band) { return PALETTE.bands[band] || PALETTE.blue; }

// ─── SHAP Chart ────────────────────────────────────────────────────────────
let shapChartInst = null;

function renderShapChart(shapData) {
  const labels = shapData.map(d => d.feature);
  const values = shapData.map(d => d.value);
  const colors = values.map(v => v > 0 ? '#ef4444' : '#2563eb');

  const ctx = document.getElementById('shapChart').getContext('2d');
  if (shapChartInst) shapChartInst.destroy();

  shapChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const v = ctx.raw;
              const dir = v > 0 ? '▲ Increases risk' : '▼ Decreases risk';
              return ` SHAP: ${v.toFixed(4)}  |  ${dir}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: '#f1f5f9' },
          border: { dash: [4, 4] },
          ticks: { font: { size: 11 } },
          title: { display: true, text: 'SHAP Value (impact on default probability)', font: { size: 11 } }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 }, padding: 4 }
        }
      }
    }
  });
}

// ─── Loan Form Submission ──────────────────────────────────────────────────
document.getElementById('loanForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn     = document.getElementById('submitBtn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('btnSpinner');
  btn.disabled = true;
  btnText.textContent = 'Assessing…';
  spinner.style.display = 'block';

  // Convert INR values → USD equivalent (PPP)
  const loanInrL    = parseFloat(document.getElementById('loan_amnt_inr').value)    || 3.5;
  const incInrL     = parseFloat(document.getElementById('annual_inc_inr').value)   || 8;
  const revolBalK   = parseFloat(document.getElementById('revol_bal_k').value)       || 100;
  const monthlyEmi  = parseFloat(document.getElementById('monthly_emi').value)       || 5000;

  // World Bank PPP factor: 1 USD ≈ 23.9 INR
  // Hard-clamped to stay within Lending Club training distribution:
  //   loan_amnt: $1,000 – $40,000   (INR: ₹0.5L – ₹9L)
  //   annual_inc: $20,000 – $150,000 (INR: ₹4L – ₹36L)
  //   revol_bal:  $0 – $100,000      (INR: ₹0 – ₹2,390K)
  const PPP_RATE = 23.9;

  const loan_amnt_raw = (loanInrL * 100000) / PPP_RATE;
  const annual_inc_raw = (incInrL * 100000) / PPP_RATE;
  const revol_bal_raw  = (revolBalK * 1000) / PPP_RATE;

  const payload = {
    loan_amnt:           Math.min(Math.max(loan_amnt_raw, 1000),  40000),
    term:                parseInt(document.getElementById('term').value),
    emp_length:          parseFloat(document.getElementById('emp_length').value)  || 5,
    home_ownership:      document.getElementById('home_ownership').value,
    annual_inc:          Math.min(Math.max(annual_inc_raw, 20000), 150000),
    purpose:             document.getElementById('purpose').value,
    dti:                 parseFloat(document.getElementById('dti').value)         || 18,
    delinq_2yrs:         parseFloat(document.getElementById('delinq_2yrs').value) || 0,
    revol_util:          parseFloat(document.getElementById('revol_util').value)   || 40,
    revol_bal:           Math.min(revol_bal_raw, 100000),
    open_acc:            parseFloat(document.getElementById('open_acc').value)    || 8,
    total_acc:           parseFloat(document.getElementById('total_acc').value)    || 15,
    pub_rec:             parseFloat(document.getElementById('pub_rec').value)      || 0,
    fico:                parseFloat(document.getElementById('fico').value)         || 700,
    credit_history_years:parseFloat(document.getElementById('credit_history_years').value) || 10,
    city_tier:           document.getElementById('city_tier').value,
    monthly_emi:         monthlyEmi,
    annual_inc_inr:      incInrL * 100000,
  };

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderDecision(data, { loanInrL, incInrL });
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Assess Credit Risk';
    spinner.style.display = 'none';
  }
});

function renderDecision(data, ctx) {
  document.getElementById('placeholder').style.display  = 'none';
  document.getElementById('decisionCard').style.display = 'block';
  document.getElementById('indiaCard').style.display    = 'block';
  document.getElementById('shapCard').style.display     = 'block';

  // ── Decision Header ──
  const prob      = data.probability;
  const approved  = data.approved;
  const riskBand  = data.risk_band;
  
  let status = 'rejected';
  if (data.stage1_reject) {
    status = 'rejected';
  } else {
    status = approved ? 'approved' : (prob < 60 ? 'review' : 'rejected');
  }

  const header  = document.getElementById('decisionHeader');
  const verdict = document.getElementById('decisionVerdict');
  const sub     = document.getElementById('decisionSub');
  const badge   = document.getElementById('decisionBadge');

  header.className  = 'decision-header ' + status;
  verdict.className = 'decision-verdict ' + status;
  badge.className   = 'decision-badge '   + status;

  if (status === 'approved') {
    verdict.textContent = '✓ Loan Recommended';
    sub.textContent     = 'Borrower meets credit risk criteria — proceed to disbursement.';
    badge.textContent   = 'APPROVE';
  } else if (status === 'review') {
    verdict.textContent = '⚑ Manual Review Required';
    sub.textContent     = 'Moderate risk detected — escalate to underwriting team.';
    badge.textContent   = 'REVIEW';
  } else {
    verdict.textContent = '✗ Loan Declined';
    sub.textContent     = 'Elevated default risk exceeds underwriting threshold.';
    badge.textContent   = 'DECLINE';
  }

  // ── Gauge ──
  const gaugeColor = prob < 35 ? PALETTE.green : prob < 60 ? PALETTE.amber : PALETTE.red;
  document.getElementById('gaugeFill').style.width      = prob + '%';
  document.getElementById('gaugeFill').style.background = gaugeColor;
  document.getElementById('gaugeValue').textContent     = prob.toFixed(1) + '%';
  document.getElementById('gaugeValue').style.color     = gaugeColor;

  // ── Metrics ──
  const f = data.features;
  const metricsRow = document.getElementById('metricsRow');
  metricsRow.innerHTML = `
    <div class="metric-box">
      <div class="metric-label">CIBIL / FICO</div>
      <div class="metric-value">${f.fico}</div>
      <div class="metric-note">${f.fico >= 720 ? '✓ Strong' : f.fico >= 680 ? '⚑ Moderate' : '✗ Weak'}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">DTI Ratio</div>
      <div class="metric-value">${f.dti}%</div>
      <div class="metric-note">${f.dti < 35 ? '✓ Acceptable' : f.dti < 45 ? '⚑ Elevated' : '✗ High'}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Revol. Utilisation</div>
      <div class="metric-value">${f.revol_util}%</div>
      <div class="metric-note">${f.revol_util < 50 ? '✓ Healthy' : f.revol_util < 75 ? '⚑ Watch' : '✗ High'}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Risk Band</div>
      <div class="metric-value" style="color:${data.risk_color}">${riskBand}</div>
      <div class="metric-note">Model segment</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Loan / Income</div>
      <div class="metric-value">${(f.loan_income_ratio * 100).toFixed(0)}%</div>
      <div class="metric-note">${f.loan_income_ratio < 0.35 ? '✓ Acceptable' : '⚑ Elevated'}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">PD Score</div>
      <div class="metric-value" style="color:${gaugeColor}">${prob.toFixed(1)}%</div>
      <div class="metric-note">Probability of Default</div>
    </div>
  `;

  // ── India Card ──
  const ind = data.indian;
  const foirClass = ind.foir < 0.50 ? 'foir-good' : ind.foir < 0.60 ? 'foir-warn' : 'foir-bad';
  const foirNote  = ind.foir < 0.50 ? '✓ Within RBI limit' : ind.foir < 0.60 ? '⚑ Borderline' : '✗ Exceeds 60% limit';
  const tierClass = ind.city_tier === 'Tier 1' ? 'tier1' : ind.city_tier === 'Tier 2' ? 'tier2' : 'tier3';

  document.getElementById('indiaGrid').innerHTML = `
    <div class="india-item">
      <div class="india-item-label">FOIR (Fixed Obligation Ratio)</div>
      <div class="india-item-value ${foirClass}">${(ind.foir * 100).toFixed(1)}%</div>
      <div class="india-item-sub">${foirNote}</div>
    </div>
    <div class="india-item">
      <div class="india-item-label">City Tier</div>
      <div class="india-item-value">${ind.city_tier}</div>
      <div class="india-item-sub"><span class="india-badge ${tierClass}">${ind.city_tier} · Risk Weight ${ind.city_tier_risk.toFixed(1)}x</span></div>
    </div>
    <div class="india-item">
      <div class="india-item-label">Loan Purpose (India Context)</div>
      <div class="india-item-value" style="font-size:12px">${ind.purpose_india}</div>
    </div>
    <div class="india-item">
      <div class="india-item-label">Priority Sector (RBI)</div>
      <div class="india-item-value">${ind.is_priority_sector ? 'Yes' : 'No'}</div>
      <div class="india-item-sub"><span class="india-badge ${ind.is_priority_sector ? 'priority' : 'non-priority'}">${ind.is_priority_sector ? 'Priority Sector Loan' : 'General Category'}</span></div>
    </div>
  `;

  // ── SHAP, Narrative, & Recommendations ──
  if (data.shap && data.shap.length > 0) {
    renderShapChart(data.shap);
    
    // Narrative
    document.getElementById('narrativeBlock').style.display = 'block';
    document.getElementById('narrativeText').innerHTML = data.narrative || '';
    
    // Recommendations
    if (data.recommendations && data.recommendations.length > 0) {
      document.getElementById('recBlock').style.display = 'block';
      const recList = document.getElementById('recList');
      recList.innerHTML = data.recommendations.map(r => `
        <div class="rec-item">
          <div class="rec-item-icon ${r.impact}">
            ${r.impact === 'high' ? '!' : (r.impact === 'medium' ? '?' : '✓')}
          </div>
          <div class="rec-item-body">
            <div class="rec-item-feature">${r.feature}</div>
            <div class="rec-item-text">${r.text}</div>
            <span class="rec-item-tag ${r.impact}">${r.impact.toUpperCase()} IMPACT</span>
          </div>
        </div>
      `).join('');
    }
  } else {
    document.getElementById('narrativeBlock').style.display = 'none';
    document.getElementById('recBlock').style.display = 'none';
  }

  // Scroll to results
  document.getElementById('decisionCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Portfolio Analytics ────────────────────────────────────────────────────
async function loadPortfolio() {
  try {
    const res  = await fetch('/api/portfolio');
    const data = await res.json();
    renderKPIs(data.kpis);
    renderStory1(data);
    renderStory2(data);
    renderStory3(data);
  } catch (e) {
    console.error('Portfolio load failed:', e);
  }
}

function renderKPIs(kpis) {
  document.getElementById('kpiStrip').innerHTML = `
    <div class="kpi-card kpi-accent">
      <div class="kpi-label">Total Loans Analysed</div>
      <div class="kpi-value">${kpis.total_loans}</div>
      <div class="kpi-sub">Historical test set portfolio</div>
    </div>
    <div class="kpi-card kpi-accent-amber">
      <div class="kpi-label">Actual NPA Rate</div>
      <div class="kpi-value">${kpis.npa_rate}</div>
      <div class="kpi-sub">Gross default ratio</div>
    </div>
    <div class="kpi-card kpi-accent-slate">
      <div class="kpi-label">Portfolio Exposure</div>
      <div class="kpi-value">${kpis.total_exposure}</div>
      <div class="kpi-sub">Total capital at risk (PPP)</div>
    </div>
    <div class="kpi-card kpi-accent-red">
      <div class="kpi-label">Expected Credit Loss (ECL)</div>
      <div class="kpi-value">${kpis.ecl_crore}</div>
      <div class="kpi-sub">Ind AS 109 modeled loss estimate</div>
    </div>
  `;
}

// Story 1 — Risk concentration
function renderStory1(data) {
  const rb    = data.risk_bands;
  const order = ['Very Low','Low','Moderate','High','Very High'];
  const vals  = order.filter(k => rb[k]).map(k => rb[k]);
  const lbls  = order.filter(k => rb[k]);
  const cols  = lbls.map(bandColor);

  // Donut — risk band distribution
  new Chart(document.getElementById('riskBandChart'), {
    type: 'doughnut',
    data: { labels: lbls, datasets: [{ data: vals, backgroundColor: cols, borderWidth: 2, borderColor: '#fff' }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '65%',
      plugins: {
        legend: { display: true, position: 'right', labels: { font: { size: 11 }, padding: 12, boxWidth: 12 } },
        tooltip: { callbacks: { label: c => ` ${c.label}: ${c.raw.toLocaleString()} loans` } }
      }
    }
  });

  // Bar — default rate by risk band
  const rbD = data.rb_default;
  new Chart(document.getElementById('rbDefaultChart'), {
    type: 'bar',
    data: {
      labels:   rbD.map(d => d.Risk_Band),
      datasets: [{ data: rbD.map(d => d.Actual_Default), backgroundColor: rbD.map(d => bandColor(d.Risk_Band)), borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' }, ticks: { callback: v => v + '%' }, title: { display: true, text: 'Default Rate %' } }
      },
      plugins: { tooltip: { callbacks: { label: c => ` Default Rate: ${c.raw}%` } } }
    }
  });

  // Bar — exposure by risk band
  const rbE = data.rb_exposure;
  new Chart(document.getElementById('rbExposureChart'), {
    type: 'bar',
    data: {
      labels:   rbE.map(d => d.Risk_Band),
      datasets: [{ data: rbE.map(d => d.exposure_cr), backgroundColor: rbE.map(d => bandColor(d.Risk_Band)), borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' }, ticks: { callback: v => '₹' + v + 'Cr' }, title: { display: true, text: 'Exposure ₹ Crore' } }
      },
      plugins: { tooltip: { callbacks: { label: c => ` Exposure: ₹${c.raw} Cr` } } }
    }
  });
}

// Story 2 — Credit quality
function renderStory2(data) {
  // FICO default
  const fd = data.fico_default;
  new Chart(document.getElementById('ficoDefaultChart'), {
    type: 'bar',
    data: {
      labels:   fd.map(d => d.fico_bucket),
      datasets: [{ data: fd.map(d => d.Actual_Default), backgroundColor: PALETTE.blue, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' }, ticks: { callback: v => v + '%' }, title: { display: true, text: 'Default Rate %' } }
      },
      plugins: { tooltip: { callbacks: { label: c => ` Default Rate: ${c.raw}%` } } }
    }
  });

  // DTI default
  const dd = data.dti_default;
  new Chart(document.getElementById('dtiDefaultChart'), {
    type: 'line',
    data: {
      labels:   dd.map(d => d.dti_bin),
      datasets: [{
        data: dd.map(d => d.Actual_Default),
        borderColor: PALETTE.amber, backgroundColor: 'rgba(217,119,6,.08)',
        tension: 0.3, pointRadius: 5, pointBackgroundColor: PALETTE.amber, fill: true
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' }, ticks: { callback: v => v + '%' }, title: { display: true, text: 'Default Rate %' } }
      },
      plugins: { tooltip: { callbacks: { label: c => ` Default Rate: ${c.raw}%` } } }
    }
  });

  // Income default
  const id = data.inc_default;
  new Chart(document.getElementById('incDefaultChart'), {
    type: 'bar',
    data: {
      labels:   id.map(d => d.inc_bin),
      datasets: [{ data: id.map(d => d.Actual_Default), backgroundColor: PALETTE.blue2, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#f1f5f9' }, ticks: { callback: v => v + '%' }, title: { display: true, text: 'Default Rate %' } }
      },
      plugins: { tooltip: { callbacks: { label: c => ` Default Rate: ${c.raw}%` } } }
    }
  });
}

// Story 3 — Product mix
function renderStory3(data) {
  // Purpose default (sorted by default rate)
  const pd = [...data.purpose_default].sort((a, b) => b.default_rate - a.default_rate);
  new Chart(document.getElementById('purposeDefaultChart'), {
    type: 'bar',
    data: {
      labels:   pd.map(d => d.purpose),
      datasets: [{ data: pd.map(d => d.default_rate), backgroundColor: PALETTE.blue, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 10 }, maxRotation: 25 } },
        y: { grid: { color: '#f1f5f9' }, ticks: { callback: v => v + '%' }, title: { display: true, text: 'Default Rate %' } }
      },
      plugins: { tooltip: { callbacks: {
        label: c => {
          const row = pd[c.dataIndex];
          return [` Default Rate: ${c.raw}%`, ` Volume: ${row.volume.toLocaleString()} loans`];
        }
      }}}
    }
  });

  // Home ownership
  const hd = data.home_default;
  new Chart(document.getElementById('homeDefaultChart'), {
    type: 'bar',
    data: {
      labels:   hd.map(d => d.home_ownership),
      datasets: [{ data: hd.map(d => d.default_rate), backgroundColor: PALETTE.blue3, borderRadius: 4 }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#f1f5f9' }, ticks: { callback: v => v + '%' }, title: { display: true, text: 'Default Rate %' } },
        y: { grid: { display: false } }
      },
      plugins: { tooltip: { callbacks: {
        label: c => {
          const row = hd[c.dataIndex];
          return [` Default Rate: ${c.raw}%`, ` Volume: ${row.volume.toLocaleString()} loans`];
        }
      }}}
    }
  });
}

// ─── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadPortfolio);
