<!-- src/App.vue -->
<template>
  <main class="wrap">
    <header class="header">
      <h1>Loan Risk Dashboard</h1>
      <small>API: {{ apiBase }}</small>
    </header>

    <!-- KPI CARDS -->
    <section class="kpis" v-if="summary">
      <div class="card">
        <div class="label">Total</div>
        <div class="value">{{ summary.total }}</div>
      </div>
      <div class="card">
        <div class="label">High Risk</div>
        <div class="value warn">{{ summary.high }}</div>
      </div>
      <div class="card">
        <div class="label">Low Risk</div>
        <div class="value">{{ summary.low }}</div>
      </div>
      <div class="card">
        <div class="label">High Rate</div>
        <div class="value">{{ (summary.high_rate * 100).toFixed(1) }}%</div>
      </div>
    </section>

    <!-- ACTIONS -->
    <section class="actions">
      <button @click="reloadAll" :disabled="loading">
        {{ loading ? 'Loading…' : 'Reload' }}
      </button>
      <button @click="seedOne" :disabled="loading || !mlOk">
        Seed 1 Prediction (RF)
      </button>
      <span v-if="!mlOk" class="hint">ML API not reachable (start FastAPI on 8000)</span>
    </section>

    <!-- RECENT TABLE -->
    <section class="table-wrap">
      <h2>Recent Predictions</h2>
      <table class="table" v-if="recent.length">
        <thead>
          <tr>
            <th>ID</th>
            <th>Created</th>
            <th>Backend</th>
            <th>Risk Prob</th>
            <th>Risk Label</th>
            <th>Credit</th>
            <th>Income</th>
            <th>Loan</th>
            <th>DTI</th>
            <th>Text Score</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in recent" :key="r.id">
            <td>{{ r.id }}</td>
            <td>{{ fmtDate(r.created_at) }}</td>
            <td>{{ r.backend }}</td>
            <td>{{ (r.risk_probability ?? 0).toFixed(3) }}</td>
            <td :class="r.risk_label === 1 ? 'danger' : 'ok'">{{ r.risk_label }}</td>
            <td>{{ r.credit_score }}</td>
            <td>{{ r.income }}</td>
            <td>{{ r.loan_amount }}</td>
            <td>{{ r.dti }}</td>
            <td>{{ r.text_risk_score ?? '-' }}</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty">
        No rows yet — send a POST to /api/predict or use “Seed 1 Prediction”.
      </div>
    </section>

    <!-- SUMMARY BY DAY -->
    <section class="table-wrap">
      <h2>Summary by Day (last 30)</h2>
      <table class="table" v-if="byDay.length">
        <thead>
          <tr>
            <th>Day</th>
            <th>Total</th>
            <th>High</th>
            <th>High Rate</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in byDay" :key="d.day">
            <td>{{ d.day }}</td>
            <td>{{ d.total }}</td>
            <td>{{ d.high }}</td>
            <td>{{ (d.high_rate * 100).toFixed(1) }}%</td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty">No daily stats yet.</div>
    </section>

    <!-- ANALYTICS (charts + cohorts) -->
    <RiskAnalytics />

    <!-- WHAT-IF SIMULATOR (live scoring) -->
    <WhatIfSimulator />
  </main>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import RiskAnalytics from './components/RiskAnalytics.vue'
import WhatIfSimulator from './components/WhatIfSimulator.vue'

type Summary = { total: number; high: number; low: number; high_rate: number }
type RecentRow = {
  id: number; created_at: string; backend: string;
  risk_probability: number; risk_label: number;
  credit_score: number; income: number; loan_amount: number; dti: number;
  text_risk_score: number | null;
}
type DayRow = { day: string; total: number; high: number; high_rate: number }

const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:5050'

const loading = ref(false)
const summary = ref<Summary | null>(null)
const recent = ref<RecentRow[]>([])
const byDay = ref<DayRow[]>([])
const mlOk = ref<boolean>(false)

const fmtDate = (iso: string) => new Date(iso).toLocaleString()

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${apiBase}${path}`)
  if (!res.ok) throw new Error(`${path} ${res.status}`)
  return res.json() as Promise<T>
}

async function reloadAll() {
  loading.value = true
  try {
    // Node health
    await getJSON('/health')

    // ML health (best-effort)
    try {
      const r = await fetch('http://127.0.0.1:8000/health')
      mlOk.value = r.ok
    } catch {
      mlOk.value = false
    }

    summary.value = await getJSON<Summary>('/api/predictions/summary')

    const recentResp = await getJSON<{ data: RecentRow[]; meta: any }>(
      '/api/predictions/recent?limit=10'
    )
    recent.value = recentResp.data

    const byDayResp = await getJSON<{ data: DayRow[] }>(
      '/api/predictions/summary-by-day'
    )
    byDay.value = byDayResp.data
  } catch (e) {
    console.error(e)
    alert('Failed to load from Node API. Is it running on 5050?')
  } finally {
    loading.value = false
  }
}

async function seedOne() {
  try {
    const body = {
      income: 25000,
      credit_score: 560,
      loan_amount: 20000,
      tenure_months: 12,
      dti: 0.6,
      application_text:
        'I had late payments last year but now I have a stable job and pay on time.'
    }
    const res = await fetch(`${apiBase}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    if (!res.ok) {
      const t = await res.text()
      throw new Error(`POST /api/predict -> ${res.status}: ${t}`)
    }
    await reloadAll()
  } catch (e) {
    console.error(e)
    alert('POST /api/predict failed. Check ML and Node logs.')
  }
}

onMounted(reloadAll)
</script>

<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial; }
.wrap { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
.header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }
.header h1 { margin: 0; font-size: 28px; }
small { opacity: 0.7; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
.card { background: #1f1f1f; border: 1px solid #333; border-radius: 12px; padding: 14px; }
.label { font-size: 12px; opacity: 0.7; }
.value { font-size: 24px; font-weight: 700; }
.value.warn { color: #ffcf5c; }
.table-wrap { background: #161616; border: 1px solid #2a2a2a; border-radius: 12px; padding: 12px; margin: 16px 0; }
.table-wrap h2 { margin: 6px 0 10px; font-size: 18px; }
.table { width: 100%; border-collapse: collapse; font-size: 14px; }
.table th, .table td { padding: 8px 10px; border-bottom: 1px solid #2a2a2a; text-align: left; white-space: nowrap; }
.table td.danger { color: #ff7f7f; font-weight: 700; }
.table td.ok { color: #8ee08e; font-weight: 700; }
.empty { opacity: 0.7; padding: 12px; }
.actions { display: flex; align-items: center; gap: 10px; margin: 8px 0 12px; }
button { background: #2d6cdf; color: white; border: none; border-radius: 8px; padding: 8px 12px; cursor: pointer; }
button:disabled { opacity: 0.6; cursor: not-allowed; }
.hint { opacity: 0.75; font-size: 12px; }
</style>
