import os
from pathlib import Path

# Paths
PROJECT_DIR = Path("C:/Users/karth/indian-insider")
ARTIFACT_PATH = Path("C:/Users/karth/.gemini/antigravity/brain/2413112e-432a-42e4-9510-5c83014566a1/entire_codebase_report.md")
LOCAL_OUTPUT_PATH = PROJECT_DIR / "entire_codebase_report.md"

# Categories
CATEGORIES = {
    "1. Shared Foundation & Database Schema": [
        "agents/common.py"
    ],
    "2. Data Collectors & Ingestion Pipeline": [
        "agents/nse_collector.py",
        "agents/fundamental_collector.py",
        "agents/event_detector.py"
    ],
    "3. Analysis & Research Agents (Scouts)": [
        "agents/doraemon.py",
        "agents/shinchan.py",
        "agents/nobita.py",
        "agents/dekisugi.py",
        "agents/suneo.py"
    ],
    "4. Core Strategy Engines & Scoring": [
        "agents/scoring_engine.py",
        "agents/canslim_engine.py",
        "agents/multibagger_engine.py",
        "agents/valuation_engine.py",
        "agents/sector_specific_metrics.py",
        "agents/management_credibility.py",
        "agents/opportunity_engine.py"
    ],
    "5. Consensus & Dispatcher (Consensus Group)": [
        "agents/doraemi.py",
        "agents/gian.py",
        "agents/investment_committee.py"
    ],
    "6. Backtesting, Optimization & Factor Validation": [
        "agents/backtester.py",
        "agents/backtest_audit.py",
        "agents/timestamp_integrity_audit.py",
        "agents/credibility_factor_test.py",
        "agents/factor_decay_test.py",
        "agents/weight_optimizer.py",
        "agents/factor_attribution.py"
    ],
    "7. Configuration Files": [
        "config/.env.example",
        "config/portfolio_current.example.json",
        "config/portfolio_current.json",
        "config/portfolio_target.example.json",
        "config/portfolio_target.json"
    ],
    "8. Installation & Deployment Scripts": [
        "install/schedule_windows.ps1",
        "install/schedule_linux.sh",
        "install/schedule_mac.sh",
        "install/uninstall_windows.ps1",
        "install/uninstall_linux.sh",
        "install/uninstall_mac.sh"
    ],
    "9. Auxiliary Analysis Scripts": [
        "scratch/calc_tstat.py",
        "scratch/check_corr.py"
    ]
}

def clean_file_content(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"Error reading file {path.name}: {e}"

def generate_markdown():
    md = []
    md.append("# 🇮🇳 Indian Insider — Full Project Codebase\n")
    md.append("This document contains the complete, untruncated source code of all active scripts, configurations, and utilities in the Indian Insider project.\n")
    
    # Generate Table of Contents
    md.append("## 📋 Table of Contents\n")
    for cat_name, files in CATEGORIES.items():
        md.append(f"### {cat_name}\n")
        for f in files:
            anchor = f.lower().replace("/", "").replace(".", "").replace(" ", "-").replace("_", "-")
            md.append(f"- [{f}](#-{anchor})\n")
        md.append("\n")
    
    md.append("---\n\n")
    
    # Append File Contents
    for cat_name, files in CATEGORIES.items():
        md.append(f"# {cat_name}\n\n")
        for f in files:
            file_path = PROJECT_DIR / f
            if not file_path.exists():
                print(f"Warning: File {f} does not exist at {file_path}")
                continue
            
            # Identify extension for code block syntax highlighting
            ext = file_path.suffix.lstrip(".")
            if ext == "py":
                lang = "python"
            elif ext == "json":
                lang = "json"
            elif ext in ["sh", "bash"]:
                lang = "bash"
            elif ext == "ps1":
                lang = "powershell"
            elif file_path.name.startswith(".env"):
                lang = "properties"
            else:
                lang = ""
            
            content = clean_file_content(file_path)
            
            # Normalize line endings to avoid double spacing in markdown
            content = content.replace("\r\n", "\n")
            
            md.append(f"## 📄 {f}\n\n")
            md.append(f"```{lang}\n")
            md.append(content)
            if not content.endswith("\n"):
                md.append("\n")
            md.append("```\n\n")
            md.append("---\n\n")
            
    # Write to files
    try:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text("".join(md), encoding="utf-8")
        print(f"Artifact updated at: {ARTIFACT_PATH}")
    except Exception as e:
        print(f"Error writing artifact: {e}")
        
    try:
        LOCAL_OUTPUT_PATH.write_text("".join(md), encoding="utf-8")
        print(f"Local file updated at: {LOCAL_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing local copy: {e}")

if __name__ == "__main__":
    generate_markdown()
