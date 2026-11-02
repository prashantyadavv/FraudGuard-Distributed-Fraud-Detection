"""Download PaySim when Kaggle credentials are available."""
import subprocess
from pathlib import Path
from src.config import DATA_DIR, ensure_directories

def main():
    """Run Kaggle CLI and provide an actionable fallback when it is not configured."""
    ensure_directories()
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "ealaxi/paysim1", "-p", str(DATA_DIR), "--unzip"], check=True)
        print("PaySim downloaded to", DATA_DIR)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Kaggle download needs kaggle.json credentials. Create an API token at kaggle.com/settings, place it in ~/.kaggle/, then retry.\nFallback: python data_generator.py --rows 100000")
if __name__ == "__main__": main()
