# StockLens

Indian equity portfolio platform with an integrated market-intelligence engine.

| Part | Where | Runs |
|------|-------|------|
| **Web app** | `src/` | Railway (browser) |
| **Engine** | `engine/` | Your PC (background) |

The engine collects NSE data, runs AI scouts, scores stocks, and pushes alerts into the web app **Alerts → Signal queue**.

---

## Quick start

### 1. Web app (local dev)

```bash
npm install
cp .env.example .env.local   # Supabase + Angel One — see DEPLOYMENT.md
npm run dev
```

Open http://localhost:3000

### 2. Engine (your PC)

**Windows:**

```powershell
git clone https://github.com/Karthik96CFA/stocklens.git
cd stocklens
powershell -ExecutionPolicy Bypass -File engine\install\setup_windows.ps1
notepad engine\.env
python engine\agents\stocklens_bridge.py --test
```

**Linux / Mac:**

```bash
pip install -r engine/requirements.txt
cp engine/config/.env.example engine/.env
python engine/agents/orchestrator.py --phase tick --dry-run
bash engine/install/schedule_linux.sh
```

### 3. Connect engine → web

In `engine/.env`:

```properties
STOCKLENS_URL=https://your-app.up.railway.app
```

---

## Engine commands

```bash
python engine/agents/orchestrator.py --phase tick
python engine/agents/orchestrator.py --phase morning --skip-trading-day-check
python engine/agents/stocklens_bridge.py --sync-opportunities
```

Or use npm shortcuts from repo root:

```bash
npm run engine:tick
npm run engine:sync
```

---

## Production

- **Web:** Railway — see [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Engine:** Windows Task Scheduler (7 phases) via `engine/install/setup_windows.ps1`

---

## Repo layout

```
stocklens/
├── src/                 Next.js app (portfolio, alerts, pool fund)
├── drizzle/             Postgres schema + migrations
├── engine/              Python intelligence engine
│   ├── agents/
│   ├── config/
│   └── install/
├── DEPLOYMENT.md
└── package.json
```

---

## Migrated from `indian-insider`

The former standalone [indian-insider](https://github.com/Karthik96CFA/indian-insider) repo now lives in `engine/`. Use this monorepo going forward.
