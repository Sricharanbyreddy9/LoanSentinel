# LoanSentinel — AI-Powered Loan Risk Prediction & Monitoring System

**Tech Stack:** FastAPI (Python) · Node.js (Express) · Vue.js (Vite) · Supabase (PostgreSQL) · NLP · PyTorch (optional)  
**Focus:** Predict, analyze, and visualize loan default risks with machine learning, natural language processing, and live dashboards.

---

## 🚀 Overview

**LoanSentinel** is a **full-stack AI system** that predicts and monitors loan risk in real time.  
It combines:

- ✅ **ML API (FastAPI + RandomForest + optional PyTorch)** — to calculate loan risk probability.  
- ✅ **Node.js API (Express)** — middleware that validates inputs, calls ML models, and stores results in Supabase.  
- ✅ **Supabase PostgreSQL Database** — to persist every prediction with input data, results, and NLP flags.  
- ✅ **Dashboard (Vue.js + Vite)** — a front-end portal to visualize KPIs, trends, and perform “what-if” simulations.  
- ✅ **NLP Layer** — performs keyword-based text risk detection from loan application statements.

---

## 🏗️ Folder Structure

```
Loan Risk System/
│
├── services/
│   ├── ml-api/                    # FastAPI ML backend (Python)
│   │   ├── main.py                # /predict, /predict_torch, /nlp endpoints
│   │   ├── torch_model.py         # Optional PyTorch model definition
│   │   └── .venv/                 # Local Python environment (ignored)
│   │
│   └── node-api/                  # Node.js Express backend
│       ├── index.js               # Main API routes (proxy + DB insert/read)
│       ├── package.json           # Node dependencies
│       ├── db-check.mjs           # Quick DB test (optional)
│       └── loan-risk-dashboard/   # Vue.js + Vite frontend
│           ├── src/               # Vue components (KPI, Tables, Charts, Simulator)
│           ├── vite.config.js     # Vite configuration
│           └── .env               # VITE_API_BASE=http://localhost:5050
│
├── models/                        # Trained ML model files
│   ├── rf_loan.joblib             # RandomForest model
│   ├── rf_loan.meta.json          # Feature metadata
│   └── risknet.pt                 # Optional PyTorch model
│
├── data/                          # Sample data for local testing
│   └── loan_data.csv
│
└── .env                           # Environment variables (local, not committed)
```
⚙️ How It Works (System Flow)
```
Vite + Vue Dashboard  →  Node.js API  →  FastAPI ML Service  →  Supabase DB
          ↑                   ↓                ↑
          └────── Reads KPIs, tables, trends ──┘
```
Frontend (Vue Dashboard)
Displays KPIs, tables, and risk trends, and allows live simulations.

Node.js Backend (Express)
Acts as the “traffic controller.” Validates data, calls ML API, and stores results.

ML API (FastAPI, Python)
Handles predictions, NLP analysis, and model operations.

Supabase Database (PostgreSQL)
Stores all inputs, outputs, and analytics-ready metadata.

🌐 API Endpoints
```
ML API (FastAPI, Port 8000)
---------------------------
GET   /health
POST  /predict
POST  /predict_torch
POST  /nlp/extract

```
Node API (Express, Port 5050)
-----------------------------
```
GET   /health
POST  /api/predict
POST  /api/predict-torch
GET   /api/predictions/recent?limit=10
GET   /api/predictions/summary
GET   /api/predictions/summary-by-day
GET   /api/predictions/summary-by-backend
```

🧩 Ports and Their Roles
Service	Tech Stack	Port	Purpose
ML API	FastAPI (Python)	8000	Runs prediction and NLP endpoints
Node API	Express (JavaScript)	5050	Middle layer between frontend & ML; handles DB writes
Dashboard	Vue (Vite)	5173	Frontend dashboard visualizing predictions
Supabase	PostgreSQL Cloud	Remote	Database to store predictions

🧪 Backend Testing with Postman
Before connecting the frontend, all backend APIs were tested individually using Postman with different ports.

Ports used during testing:

scss
Copy code
FastAPI (ML API)  →  8000  
Node.js (Express) →  5050  
Frontend (Vue)    →  5173
✅ Step-by-step Postman Tests
1️⃣ FastAPI ML API (Port 8000)

GET http://127.0.0.1:8000/health → verifies ML API is live.

POST http://127.0.0.1:8000/predict → tests loan prediction.

POST http://127.0.0.1:8000/nlp/extract → tests NLP keyword scoring.

✅ Result: Machine learning model and NLP pipeline validated independently.

2️⃣ Node.js API (Port 5050)

GET http://localhost:5050/health → verifies Node service.

POST http://localhost:5050/api/predict → forwards data to ML API, inserts into Supabase.

GET http://localhost:5050/api/predictions/recent → fetches saved predictions.

GET http://localhost:5050/api/predictions/summary → verifies KPI summaries.

✅ Result: End-to-end backend data flow (ML → Node → Supabase) confirmed.

3️⃣ Integration Test
After backend tests, the frontend (port 5173) was connected via:

ini
Copy code
VITE_API_BASE=http://localhost:5050
✅ Final Result: Postman-validated endpoints were displayed live in the dashboard.

🖥️ Frontend Integration (Vue + Vite)
Fetches data from Node APIs:
/api/predictions/summary, /api/predictions/recent, /api/predictions/summary-by-day.

Displays results visually using cards, tables, and charts.

Supports “what-if” risk simulation: enter data → auto-predict → instant UI update.

📊 Dashboard Features
KPI Cards: Total, High Risk, Low Risk, High-Risk %

Recent Predictions: Table with recent loan scores

Summary by Day: Daily risk trend visualization

Risk Analytics: Histogram & cohort-based grouping

What-If Simulator: Real-time scoring on custom inputs

💾 Database Schema
sql
Copy code
CREATE TABLE IF NOT EXISTS public.predictions (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  income NUMERIC,
  credit_score NUMERIC,
  loan_amount NUMERIC,
  tenure_months INT,
  dti NUMERIC,
  risk_probability NUMERIC,
  risk_label INT,
  backend TEXT,                 -- 'rf' or 'torch'
  application_text TEXT,        -- optional text input
  nlp_flags JSONB,              -- NLP keyword hits
  text_risk_score NUMERIC       -- text-based risk score
);
🧠 Features Implemented
markdown
Copy code
| Category | Description |
|-----------|--------------|
| **Machine Learning** | RandomForest model with optional PyTorch backend. |
| **NLP Layer** | Keyword-based analysis flags risky words in applications. |
| **API System** | REST APIs for predictions and analytics. |
| **Database** | Supabase PostgreSQL for persistent prediction logs. |
| **Frontend** | Vue + Vite dashboard visualizing KPIs and charts. |
| **Testing** | Full Postman validation before frontend connection. |
| **Integration** | ML API ↔ Node.js ↔ Supabase ↔ Vue fully synchronized. |

🧪 How to Run Locally
---
1️⃣ Run ML API
```
bash
Copy code
cd "services/ml-api"
uvicorn main:app --reload --port 8000
➡️ Open: http://127.0.0.1:8000
```
2️⃣ Run Node API

```
cd "services/node-api"
npm install
npm run dev
➡️ Open: http://localhost:5050
```
3️⃣ Run Dashboard

```
cd "services/node-api/loan-risk-dashboard"
echo VITE_API_BASE=http://localhost:5050 > .env
npm install
npm run dev
➡️ Open: http://localhost:5173
```

🔒 Example .env Configuration
```
# Backend and API Ports
PORT_NODE=5050
PORT_ML=8000
CORS_ORIGIN=http://localhost:5173

# Supabase PostgreSQL Connection
PGHOST=db.example.supabase.co
PGPORT=6543
PGDATABASE=postgres
PGUSER=postgres
PGPASSWORD=yourpassword

# PostgreSQL Connection URL (URL-encoded if needed)
DATABASE_URL=postgresql://postgres:yourpassword@db.example.supabase.co:5432/postgres?sslmode=require

# ML API Base URL (used by Node.js to call FastAPI)
ML_BASE=http://127.0.0.1:8000
```
🧭 Future Improvements
Integrate Hugging Face / spaCy NLP pipelines for deeper analysis.

Add feature importance visualization (SHAP).

Create batch scoring for large datasets (80K+ loans).

Enable Supabase Auth for user-based analytics.

Add Docker Compose for one-click environment setup.

Implement alert notifications for daily high-risk spikes.

👨‍💻 Project Summary (Beginner Friendly)
This project connects:

A web dashboard (Vue) for loan entry and visualization.

A Node.js API that handles logic and database writes.

A FastAPI service that predicts loan risk and analyzes text.

A Supabase database that stores results for analytics.

In short — it’s a complete AI + full-stack workflow showing how to link
machine learning, backend APIs, databases, and a real-time frontend.

🧾 Author
Sri Charan Byreddy (Ranadeep Mahendra)
Full Stack Software Engineer | AI, ML, and Cloud Systems
---

✅ **Instructions to add this:**

1. Open your project folder.  
2. Create or replace your `README.md` file.  
3. Paste everything above (exactly as-is).  
4. Commit and push:
   ```bash
   git add README.md
   git commit -m "Added final complete project README with backend, frontend, and Postman details"
   git push
