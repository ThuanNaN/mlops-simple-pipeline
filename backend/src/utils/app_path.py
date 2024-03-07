from pathlib import Path

class AppPath:
    ROOT_DIR = Path(__file__).parent.parent.parent

    SOUCE_DIR = ROOT_DIR / "src"

    CONFIG_DIR = SOUCE_DIR / "config"
    SERVE_CONFIG = CONFIG_DIR / "serve_config"
    DATA_CONFIG = CONFIG_DIR / "data_config"

    DATA_DIR = ROOT_DIR / "data_source"
    RAW_DATA_DIR = DATA_DIR / "catdog_raw"
    COLLECTED_DATA_DIR = DATA_DIR / "collected"
    TRAIN_DATA_DIR = DATA_DIR / "train_data"


AppPath.COLLECTED_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

