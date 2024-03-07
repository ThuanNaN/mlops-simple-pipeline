from pathlib import Path

class AppPath:
    ROOT_DIR = Path(__file__).parent.parent
    
    LOG_DIR = ROOT_DIR / "logs"
    
    CACHE_DIR = ROOT_DIR / "cache"
    CAPTURED_DATA_DIR = CACHE_DIR / "captured_data"

AppPath.LOG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CACHE_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)


