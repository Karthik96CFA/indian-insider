
---

## Indian Insider Engine (PC)

The Python market-intelligence engine lives in `engine/` and runs on **your PC** (not Railway). It pushes signals into StockLens via `POST /api/integrations/signals`.

### Setup on Windows

```powershell
git clone https://github.com/Karthik96CFA/stocklens.git
cd stocklens
powershell -ExecutionPolicy Bypass -File engine\install\setup_windows.ps1
notepad engine\.env
python engine\agents\stocklens_bridge.py --test
```

### Engine environment (`engine/.env`)

| Variable | Purpose |
| :--- | :--- |
| `GEMINI_API_KEY` | Google AI Studio key for scout agents |
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` / `GMAIL_TO` | Email alerts (use a Google **App Password**) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Optional Telegram delivery |
| `STOCKLENS_URL` | Your Railway app URL |
| `STOCKLENS_INTEGRATION_SECRET` | Must match Railway `INTEGRATION_SECRET` when set |

Signals appear in **Alerts -> Signal queue** in the web app.
