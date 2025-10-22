// services/node-api/index.js
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fetch from "node-fetch";
import pkg from "pg";
const { Pool } = pkg;

/* -------------------- env + app -------------------- */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const app = express();
app.use(express.json());
app.use(cors({ origin: process.env.CORS_ORIGIN || "http://localhost:5173" }));
/* -------------------- database pool -------------------- */
const pool = new Pool({
  host: process.env.PGHOST,
  port: Number(process.env.PGPORT || 6543),
  database: process.env.PGDATABASE,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  ssl: { require: true, rejectUnauthorized: false },
});

/* -------------------- helpers -------------------- */
const ML_BASE = process.env.ML_BASE || "http://127.0.0.1:8000";

function basicEdgeValidate(body) {
  const { income, credit_score, loan_amount, tenure_months, dti } = body ?? {};
  if (typeof credit_score !== "number" || credit_score < 0 || credit_score > 850) {
    return { field: "credit_score", message: "credit_score must be a number between 0 and 850" };
  }
  if (typeof income !== "number" || income < 0) {
    return { field: "income", message: "income must be a non-negative number" };
  }
  if (typeof loan_amount !== "number" || loan_amount <= 0) {
    return { field: "loan_amount", message: "loan_amount must be a positive number" };
  }
  if (!Number.isInteger(tenure_months) || tenure_months <= 0) {
    return { field: "tenure_months", message: "tenure_months must be a positive integer" };
  }
  if (typeof dti !== "number" || dti < 0 || dti > 10) {
    return { field: "dti", message: "dti must be a number between 0 and 10" };
  }
  return null;
}

/** Prefer /nlp/score, fallback to /nlp/extract. Returns { textRisk, flags, appText } */
async function getNlpAugmentation(rawText) {
  const appText = (rawText ?? "").trim();
  if (!appText) return { textRisk: null, flags: null, appText: null };

  // 1) Try transformer score
  try {
    const s = await fetch(`${ML_BASE}/nlp/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ application_text: appText }),
    });
    if (s.ok) {
      const js = await s.json();
      return { textRisk: js.text_risk_score ?? null, flags: null, appText };
    }
  } catch (e) {
    console.warn("NLP /nlp/score error:", e);
  }

  // 2) Fallback to keyword extract
  try {
    const e = await fetch(`${ML_BASE}/nlp/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ application_text: appText }),
    });
    if (e.ok) {
      const js = await e.json();
      return { textRisk: js.text_risk_score ?? null, flags: js.flags ?? null, appText };
    }
  } catch (err) {
    console.warn("NLP /nlp/extract error:", err);
  }

  return { textRisk: null, flags: null, appText };
}

/* -------------------- health -------------------- */
app.get("/health", (_req, res) => res.json({ status: "ok" }));

/* -------------------- RF prediction -------------------- */
app.post("/api/predict", async (req, res) => {
  try {
    const err = basicEdgeValidate(req.body);
    if (err) return res.status(422).json({ error: "validation_error", ...err });

    const mlUrl = `${ML_BASE}/predict`;
    console.log("Calling ML RF:", mlUrl);

    const r = await fetch(mlUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });

    if (!r.ok) {
      const text = await r.text();
      const status = r.status;
      console.error("ML RF error:", status, text);
      const forwardStatus = status >= 400 && status < 500 ? status : 502;
      return res.status(forwardStatus).json({ error: "ml_api_error", status, details: text });
    }

    const out = await r.json();

    // optional NLP enrichment
    const { textRisk, flags, appText } = await getNlpAugmentation(req.body.application_text);

    const { income, credit_score, loan_amount, tenure_months, dti } = req.body;
    const sql = `
      INSERT INTO public.predictions
      (income, credit_score, loan_amount, tenure_months, dti,
       risk_probability, risk_label, backend,
       application_text, nlp_flags, text_risk_score)
      VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
      RETURNING id, created_at;
    `;
    const vals = [
      income,
      credit_score,
      loan_amount,
      tenure_months,
      dti,
      out.risk_probability,
      out.risk_label,
      "rf",
      appText,
      flags ? JSON.stringify(flags) : null,
      textRisk,
    ];

    const ins = await pool.query(sql, vals);
    const inserted = ins.rows[0];

    res.json({
      ...out,
      id: inserted?.id,
      created_at: inserted?.created_at,
      backend: "rf",
      text_risk_score: textRisk,
    });
  } catch (e) {
    console.error("Server error (rf):", e);
    res.status(500).json({ error: "server_error", message: e.message });
  }
});

/* -------------------- Torch prediction -------------------- */
app.post("/api/predict-torch", async (req, res) => {
  try {
    const err = basicEdgeValidate(req.body);
    if (err) return res.status(422).json({ error: "validation_error", ...err });

    const mlUrl = `${ML_BASE}/predict_torch`;
    console.log("Calling ML Torch:", mlUrl);

    const r = await fetch(mlUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });

    if (!r.ok) {
      const text = await r.text();
      const status = r.status;
      console.error("ML Torch error:", status, text);
      const forwardStatus = status >= 400 && status < 500 ? status : 502;
      return res.status(forwardStatus).json({ error: "ml_api_error", status, details: text });
    }

    const out = await r.json();

    // optional NLP enrichment
    const { textRisk, flags, appText } = await getNlpAugmentation(req.body.application_text);

    const { income, credit_score, loan_amount, tenure_months, dti } = req.body;
    const sql = `
      INSERT INTO public.predictions
      (income, credit_score, loan_amount, tenure_months, dti,
       risk_probability, risk_label, backend,
       application_text, nlp_flags, text_risk_score)
      VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
      RETURNING id, created_at;
    `;
    const vals = [
      income,
      credit_score,
      loan_amount,
      tenure_months,
      dti,
      out.risk_probability,
      out.risk_label,
      "torch",
      appText,
      flags ? JSON.stringify(flags) : null,
      textRisk,
    ];

    const ins = await pool.query(sql, vals);
    const inserted = ins.rows[0];

    res.json({
      ...out,
      id: inserted?.id,
      created_at: inserted?.created_at,
      backend: "torch",
      text_risk_score: textRisk,
    });
  } catch (e) {
    console.error("Server error (torch):", e);
    res.status(500).json({ error: "server_error", message: e.message });
  }
});

/* -------------------- reads -------------------- */
// latest rows (paginated)
app.get("/api/predictions/recent", async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit ?? "10", 10), 100);
    const offset = Math.max(parseInt(req.query.offset ?? "0", 10), 0);

    const sql = `
      SELECT
        id,
        created_at,
        CAST(income           AS double precision) AS income,
        CAST(credit_score     AS double precision) AS credit_score,
        CAST(loan_amount      AS double precision) AS loan_amount,
        tenure_months,
        CAST(dti              AS double precision) AS dti,
        CAST(risk_probability AS double precision) AS risk_probability,
        risk_label,
        backend,
        CAST(text_risk_score  AS double precision) AS text_risk_score,
        nlp_flags,
        application_text
      FROM public.predictions
      WHERE credit_score BETWEEN 0 AND 850
      ORDER BY created_at DESC
      LIMIT $1 OFFSET $2;
    `;

    const { rows } = await pool.query(sql, [limit, offset]);
    res.json({ data: rows, meta: { limit, offset, count: rows.length } });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server_error", message: e.message });
  }
});

// summary (totals, high, low, high_rate)
app.get("/api/predictions/summary", async (_req, res) => {
  try {
    const sql = `
      SELECT
        COUNT(*)::int AS total,
        COALESCE(SUM((risk_label = 1)::int), 0)::int AS high,
        COALESCE(SUM((risk_label = 0)::int), 0)::int AS low,
        ROUND(AVG((risk_label = 1)::int)::numeric, 4) AS high_rate
      FROM public.predictions
      WHERE credit_score BETWEEN 0 AND 850;
    `;
    const { rows } = await pool.query(sql);
    res.json(rows[0]);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server_error", message: e.message });
  }
});

// summary by backend
app.get("/api/predictions/summary-by-backend", async (_req, res) => {
  try {
    const sql = `
      SELECT
        backend,
        COUNT(*)::int AS total,
        COALESCE(SUM((risk_label = 1)::int), 0)::int AS high,
        COALESCE(SUM((risk_label = 0)::int), 0)::int AS low,
        ROUND(AVG((risk_label = 1)::int)::numeric, 4) AS high_rate
      FROM public.predictions
      WHERE credit_score BETWEEN 0 AND 850
      GROUP BY backend
      ORDER BY backend;
    `;
    const { rows } = await pool.query(sql);
    res.json({ data: rows });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server_error", message: e.message });
  }
});

// daily rollup for charts (last 30 days)
app.get("/api/predictions/summary-by-day", async (_req, res) => {
  try {
    const sql = `
      SELECT
        DATE_TRUNC('day', created_at)::date AS day,
        COUNT(*)::int AS total,
        COALESCE(SUM((risk_label = 1)::int), 0)::int AS high,
        ROUND(AVG((risk_label = 1)::int)::numeric, 4) AS high_rate
      FROM public.predictions
      WHERE credit_score BETWEEN 0 AND 850
      GROUP BY day
      ORDER BY day DESC
      LIMIT 30;
    `;
    const { rows } = await pool.query(sql);
    res.json({ data: rows });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "server_error", message: e.message });
  }
});

/* -------------------- start -------------------- */
const PORT = Number(process.env.PORT_NODE || 5050);
app.listen(PORT, "0.0.0.0", () =>
  console.log(`Node API on http://localhost:${PORT}`)
);
