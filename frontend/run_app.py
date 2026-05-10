
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR  = FRONTEND_DIR.parent
APP_PATH     = FRONTEND_DIR / "app.py"


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
    tmp_dir = Path(tempfile.mkdtemp()) / ".streamlit"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_dir / "config.toml"
    config_file.write_text(STREAMLIT_CONFIG, encoding="utf-8")
    return tmp_dir.parent  # parent dir holding .streamlit/


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

    if not APP_PATH.exists():
        print(f"\n[ERROR] app.py not found at:\n  {APP_PATH}\n")
        sys.exit(1)

    try:
        import streamlit
    except ImportError:
        print("\n[ERROR] Streamlit is not installed.")
        print("  Run:  pip install -r requirements.txt\n")
        sys.exit(1)

    config_dir = _write_temp_config()
    env = os.environ.copy()
    env["STREAMLIT_CONFIG_DIR"] = str(config_dir / ".streamlit")  # inject config

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(APP_PATH),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--theme.base", "dark",
    ]
    if not args.no_browser:
        cmd += ["--server.headless", "false"]  # auto-open browser

    print("=" * 60)
    print("  Drowsiness Detection System")
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  App:  {APP_PATH}")
    print("=" * 60)
    print("  Press Ctrl+C to stop\n")

    try:
        subprocess.run(cmd, env=env, cwd=str(PROJECT_DIR), check=True)  # launch Streamlit
    except KeyboardInterrupt:
        print("\n\n  [INFO] Server stopped by user.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Streamlit exited with code {e.returncode}\n")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
