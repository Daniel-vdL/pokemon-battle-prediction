"""
Quick launcher for the Pokemon Battle Predictor GUI.
Run this from the project root: python run_gui.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / "scripts" / "gui_app.py"
    subprocess.run([sys.executable, str(script_path)])
