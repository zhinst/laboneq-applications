import configparser
from pathlib import Path


def get_config():
    config = configparser.ConfigParser()
    settings_path = Path(__file__).resolve().parent.parent / "settings.ini"
    config.read(settings_path)
    return config
