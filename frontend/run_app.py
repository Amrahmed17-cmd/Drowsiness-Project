"""
╔══════════════════════════════════════════════════════════╗
║       DrowsinessDetection — run_app.py                  ║
║       Launch the Streamlit GUI from any working dir     ║
╚══════════════════════════════════════════════════════════╝

Usage
-----
    python frontend/run_app.py               # default port 8501
    python frontend/run_app.py --port 8888   # custom port
    python frontend/run_app.py --no-browser  # headless / server mode

What this script does
---------------------
  1. Resolves the absolute path to frontend/app.py
  2. Generates a temporary Streamlit config so the UI looks
     exactly right (dark theme, wide layout, no menu bar)
  3. Calls `streamlit run` via subprocess from the project root,
     so the GUI can import `src.realtime.pipeline` without any
     extra PYTHONPATH gymnastics.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR  = FRONTEND_DIR.parent
APP_PATH     = FRONTEND_DIR / "app.py"


# ── Streamlit config (injected via env, no .streamlit/config.toml needed) ────
STREAMLIT_CONFIG = textwrap.dedent("""
    [global]
    developmentMode = false

    [server]
    headless = true
    runOnSave = false
    fileWatcherType = "none"

    [browser]
    gatherUsageStats = false

    [theme]
    base = "dark"
    backgroundColor     = "#03050e"
    secondaryBackgroundColor = "#080d1c"
    primaryColor        = "#00d4ff"
    textColor           = "#d8e8ff"
    font                = "sans serif"
""")


def _write_temp_config() -> Path:
    """Write Streamlit config to a temp file and return its directory."""
    tmp_dir = Path(tempfile.mkdtemp()) / ".streamlit"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_dir / "config.toml"
    config_file.write_text(STREAMLIT_CONFIG, encoding="utf-8")
    return tmp_dir.parent   # the dir that contains .streamlit/


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Drowsiness Detection GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python scripts/run_app.py
          python scripts/run_app.py --port 8888
          python scripts/run_app.py --no-browser
        """),
    )
    parser.add_argument("--port",       type=int, default=8501, help="Streamlit server port (default: 8501)")
    parser.add_argument("--no-browser", action="store_true",    help="Do not open browser automatically")
    args = parser.parse_args()

    # ── Pre-flight checks ───────────────────────────────────────────────────
    if not APP_PATH.exists():
        print(f"\n[ERROR] app.py not found at:\n  {APP_PATH}\n")
        sys.exit(1)

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("\n[ERROR] Streamlit is not installed.")
        print("  Run:  pip install -r requirements.txt\n")
        sys.exit(1)

    # ── Config ─────────────────────────────────────────────────────────────
    config_dir = _write_temp_config()
    env = os.environ.copy()
    env["STREAMLIT_CONFIG_DIR"] = str(config_dir / ".streamlit")

    # ── Build command ───────────────────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_PATH),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--theme.base", "dark",
    ]
    if not args.no_browser:
        cmd += ["--server.headless", "false"]

    # ── Launch ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Drowsiness Detection System")
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  App:  {APP_PATH}")
    print("=" * 60)
    print("  Press Ctrl+C to stop\n")

    try:
        subprocess.run(cmd, env=env, cwd=str(PROJECT_DIR), check=True)
    except KeyboardInterrupt:
        print("\n\n  [INFO] Server stopped by user.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Streamlit exited with code {e.returncode}\n")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()