from pathlib import Path

# Path to HDV/ folder
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = ROOT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
CONF_DIR = SRC_DIR / "configs"
DEFAULT_CONF = CONF_DIR / "config_cfg.py"