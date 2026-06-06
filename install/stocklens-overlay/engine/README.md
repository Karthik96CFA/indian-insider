# StockLens Engine (Indian Insider)

Python market-intelligence engine for StockLens. Runs on your PC and pushes signals into the StockLens web app.

## Quick start (Windows)

```powershell
cd C:\Users\karth\stocklens
powershell -ExecutionPolicy Bypass -File engine\install\setup_windows.ps1
notepad engine\.env
python engine\agents\stocklens_bridge.py --test
python engine\agents\orchestrator.py --phase tick
```

## Config (`engine/.env`)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | AI scouts |
| `GMAIL_*` | Email alerts |
| `TELEGRAM_*` | Telegram |
| `STOCKLENS_URL` | Your Railway app URL |
| `ANGELONE_*` | Portfolio drift (optional) |

## Layout

```
engine/
  agents/          orchestrator, scouts, collectors, factor engines
  config/          portfolio JSON templates
  install/         Windows / Linux / Mac schedulers
  .state/          SQLite DB + logs (created at runtime)
```

Signals appear in StockLens → **Alerts → Signal queue**.
