<template>
  <section class="analytics">
    <h2>Analytics</h2>

    <div class="grid">
      <div class="panel">
        <h3>Daily High-Risk Rate</h3>
        <canvas ref="trendRef"></canvas>
      </div>

      <div class="panel">
        <h3>Risk Probability Distribution (last 100)</h3>
        <canvas ref="histRef"></canvas>
      </div>
    </div>

    <div class="panel">
      <h3>Cohorts (last 100)</h3>
      <div class="cohorts">
        <div>
          <h4>By Credit Score</h4>
          <table>
            <thead><tr><th>Band</th><th>N</th><th>High %</th><th>Avg Prob</th></tr></thead>
            <tbody>
              <tr v-for="c in creditBands" :key="c.name">
                <td>{{ c.name }}</td>
                <td>{{ c.n }}</td>
                <td>{{ (c.highRate*100).toFixed(1) }}%</td>
                <td>{{ c.avgProb.toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          <h4>By DTI</h4>
          <table>
            <thead><tr><th>Band</th><th>N</th><th>High %</th><th>Avg Prob</th></tr></thead>
            <tbody>
              <tr v-for="c in dtiBands" :key="c.name">
                <td>{{ c.name }}</td>
                <td>{{ c.n }}</td>
                <td>{{ (c.highRate*100).toFixed(1) }}%</td>
                <td>{{ c.avgProb.toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          <h4>Backend Comparison</h4>
          <table>
            <thead><tr><th>Backend</th><th>N</th><th>High %</th><th>Avg Prob</th></tr></thead>
            <tbody>
              <tr v-for="c in backendBands" :key="c.name">
                <td>{{ c.name }}</td>
                <td>{{ c.n }}</td>
                <td>{{ (c.highRate*100).toFixed(1) }}%</td>
                <td>{{ c.avgProb.toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import {
  Chart, LineController, LineElement, PointElement, LinearScale, CategoryScale,
  BarController, BarElement, Tooltip, Legend
} from 'chart.js'

Chart.register(
  LineController, LineElement, PointElement, LinearScale, CategoryScale,
  BarController, BarElement, Tooltip, Legend
)

type DayRow = { day: string; total: number; high: number; high_rate: number }
type RecentRow = {
  id: number; created_at: string; backend: string;
  risk_probability: number; risk_label: number;
  credit_score: number; income: number; loan_amount: number; dti: number;
}

const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:5050'

const trendRef = ref<HTMLCanvasElement | null>(null)
const histRef = ref<HTMLCanvasElement | null>(null)
let trendChart: Chart | null = null
let histChart: Chart | null = null

const recent = ref<RecentRow[]>([])
const byDay = ref<DayRow[]>([])

function bucketize(rows: RecentRow[], key: (r: RecentRow)=>string) {
  const map = new Map<string, RecentRow[]>()
  for (const r of rows) {
    const k = key(r)
    map.set(k, (map.get(k) || []).concat([r]))
  }
  return [...map.entries()].map(([name, arr]) => {
    const n = arr.length
    const high = arr.filter(a => a.risk_label === 1).length
    const avgProb = arr.reduce((s,a)=>s+(a.risk_probability||0),0)/Math.max(1,n)
    return { name, n, highRate: high/Math.max(1,n), avgProb }
  }).sort((a,b)=>a.name.localeCompare(b.name))
}

const creditBands = computed(()=> bucketize(recent.value, r=>{
  const cs = r.credit_score || 0
  if (cs < 580) return '<580 (Poor)'
  if (cs < 670) return '580–669 (Fair)'
  if (cs < 740) return '670–739 (Good)'
  if (cs < 800) return '740–799 (Very Good)'
  return '800+ (Excellent)'
}))

const dtiBands = computed(()=> bucketize(recent.value, r=>{
  const d = r.dti || 0
  if (d < 0.2) return 'DTI < 0.2'
  if (d < 0.35) return '0.2–0.35'
  if (d < 0.5) return '0.35–0.5'
  if (d < 0.75) return '0.5–0.75'
  return '0.75+'
}))

const backendBands = computed(()=> bucketize(recent.value, r=> (r.backend || 'unknown')))

async function loadData() {
  const dayRes = await fetch(`${apiBase}/api/predictions/summary-by-day`)
  const dayJson = await dayRes.json() as { data: DayRow[] }
  byDay.value = dayJson.data

  const recRes = await fetch(`${apiBase}/api/predictions/recent?limit=100`)
  const recJson = await recRes.json() as { data: RecentRow[] }
  recent.value = recJson.data

  drawTrend()
  drawHistogram()
}

function drawTrend() {
  const labels = byDay.value.slice().reverse().map(d=>d.day)
  const data = byDay.value.slice().reverse().map(d=> Math.round(d.high_rate*1000)/10)
  if (trendChart) trendChart.destroy()
  trendChart = new Chart(trendRef.value as HTMLCanvasElement, {
    type: 'line',
    data: { labels, datasets: [{ label: 'High-risk % per day', data }] },
    options: { responsive: true, plugins: { legend: { display: true }, tooltip: { enabled: true } } }
  })
}

function drawHistogram() {
  const probs = recent.value.map(r=> Number(r.risk_probability||0))
  const bins = Array(10).fill(0)
  probs.forEach(p => {
    const idx = Math.min(9, Math.floor(p*10))
    bins[idx]++
  })
  const labels = ['0–0.1','0.1–0.2','0.2–0.3','0.3–0.4','0.4–0.5','0.5–0.6','0.6–0.7','0.7–0.8','0.8–0.9','0.9–1.0']
  if (histChart) histChart.destroy()
  histChart = new Chart(histRef.value as HTMLCanvasElement, {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Count', data: bins }] },
    options: { responsive: true, plugins: { legend: { display: false }, tooltip: { enabled: true } } }
  })
}

onMounted(loadData)
</script>

<style scoped>
.analytics { margin-top: 16px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel { background:#161616; border:1px solid #2a2a2a; border-radius:12px; padding:12px; }
.cohorts { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
table { width:100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #2a2a2a; white-space: nowrap; }
h3 { margin: 6px 0 10px; }
h4 { margin: 0 0 8px; }
@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; }
  .cohorts { grid-template-columns: 1fr; }
}
</style>
