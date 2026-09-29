"""
AquaProtect-AI: Centralized Configuration Loader
Loads and validates config.yaml with defensive fallback to defaults.
"""

import os
import yaml
from typing import Dict, Any

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

class ConfigLoader:
    """Singleton loader for config.yaml."""
    _instance = None
    _config = None

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        if cls._config is None:
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                        cls._config = yaml.safe_load(f)
                except Exception:
                    cls._config = {}
            else:
                cls._config = {}
        return cls._config

    @classmethod
    def get(cls, section: str, key: str, default: Any = None) -> Any:
        cfg = cls.get_config()
        return cfg.get(section, {}).get(key, default)