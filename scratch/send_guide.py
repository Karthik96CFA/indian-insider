import sys
from pathlib import Path

# Add agents directory to path
sys.path.insert(0, str(Path("C:/Users/karth/indian-insider/agents")))

from daily_briefing import _send_telegram

msg = """<b>🌅 Indian Insider — Daily Pipeline Guide</b>

<b>1. Evening Routine (Run once daily at 18:30 IST)</b>
<code>cd C:\\Users\\karth\\indian-insider\\agents
python nse_collector.py
python event_detector.py
python opportunity_engine.py</code>

<b>2. Morning Routine (Run at 08:00 IST)</b>
<code>python daily_briefing.py --dry-run   # preview
python daily_briefing.py             # send to Telegram</code>

<b>3. Paper Trading Commands</b>
<code>python paper_trading.py add DIXON BUY 15 4820.50 --reason "3-scout consensus"
python paper_trading.py view         # view live portfolio</code>

<i>Tap/click any code block to copy the command.</i>"""

_send_telegram(msg)
print("Telegram message sent successfully.")
