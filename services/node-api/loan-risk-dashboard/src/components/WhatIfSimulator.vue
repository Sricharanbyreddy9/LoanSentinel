<template>
  <section class="panel">
    <h3>What-If Risk Simulator (RF)</h3>

    <div class="grid">
      <div class="field"><label>Income</label><input type="number" v-model.number="form.income" /></div>
      <div class="field"><label>Credit Score</label><input type="number" v-model.number="form.credit_score" min="0" max="850" /></div>
      <div class="field"><label>Loan Amount</label><input type="number" v-model.number="form.loan_amount" /></div>
      <div class="field"><label>Tenure (months)</label><input type="number" v-model.number="form.tenure_months" /></div>
      <div class="field"><label>DTI</label><input type="number" step="0.01" v-model.number="form.dti" /></div>
    </div>

    <label>Application Text (optional)</label>
    <textarea v-model="form.application_text" rows="3" placeholder="e.g., had late payments in 2023 but improved in 2024"></textarea>

    <div class="actions">
      <button @click="run" :disabled="loading">{{ loading ? 'Scoring…' : 'Score with RF' }}</button>
      <span v-if="error" class="err">{{ error }}</span>
    </div>

    <div v-if="result" class="result">
      <div>Risk Probability: <b>{{ result.risk_probability.toFixed(3) }}</b></div>
      <div>Risk Label: <b :class="result.risk_label ? 'danger' : 'ok'">{{ result.risk_label }}</b></div>
      <div v-if="result.text_risk_score !== undefined">Text Risk Score: <b>{{ result.text_risk_score?.toFixed?.(3) ?? '-' }}</b></div>
      <small>Note: this also writes to the DB (like a real request).</small>
    </div>
  </section>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:5050'

const form = reactive({
  income: 25000,
  credit_score: 560,
  loan_amount: 20000,
  tenure_months: 12,
  dti: 0.6,
  application_text: 'I had late payments last year but now I have a stable job and pay on time.'
})

const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<any>(null)

async function run() {
  error.value = null
  result.value = null
  loading.value = true
  try {
    const r = await fetch(`${apiBase}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form)
    })
    const js = await r.json()
    if (!r.ok) throw new Error(js?.error || `HTTP ${r.status}`)
    result.value = js
  } catch (e:any) {
    error.value = e.message || String(e)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.panel { background:#161616; border:1px solid #2a2a2a; border-radius:12px; padding:12px; margin-top:16px; }
.grid { display:grid; grid-template-columns: repeat(3,1fr); gap:10px; margin-bottom:10px; }
.field { display:flex; flex-direction:column; gap:6px; }
input, textarea { background:#0f0f0f; border:1px solid #2a2a2a; color:#fff; border-radius:8px; padding:8px; }
.actions { margin-top:10px; display:flex; align-items:center; gap:12px; }
button { background:#2d6cdf; color:white; border:none; border-radius:8px; padding:8px 12px; cursor:pointer; }
.err { color:#ff8181; }
.result { margin-top:10px; }
.danger { color:#ff7f7f; }
.ok { color:#8ee08e; }
@media (max-width:900px){ .grid{ grid-template-columns: 1fr; } }
</style>
