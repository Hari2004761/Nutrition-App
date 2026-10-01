"""Settings from src/.env, shared by every entry point (Flask, Streamlit, CLI)."""
import os
from pathlib import Path

from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().with_name(".env")


def get_setting(name):
    """The value of `name`, or None when it is missing or blank.

    A real environment variable wins (that is how production sets it);
    otherwise src/.env is re-read on every call, so a key added while the app
    is running is picked up without a restart.
    """
    value = os.environ.get(name)
    if value is None:
        value = dotenv_values(ENV_FILE).get(name)
    return (value or "").strip() or None
