import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import pkg from "pg";

const { Pool } = pkg;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const pool = new Pool({
  host: process.env.PGHOST,
  port: Number(process.env.PGPORT || 6543),
  database: process.env.PGDATABASE,
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  ssl: { require: true, rejectUnauthorized: false },
});

try {
  const { rows } = await pool.query("SELECT NOW() as now");
  console.log("✅ Supabase connection OK. Server time:", rows[0].now);
} catch (err) {
  console.error("❌ Supabase connection failed:", err.message);
} finally {
  await pool.end();
}
