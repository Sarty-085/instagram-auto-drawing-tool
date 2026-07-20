"""Configuration management for the Instagram auto-drawing bot.

Handles loading, saving, and merging of JSON configuration files.
Supports both normal script execution and PyInstaller-frozen executables.
"""

import json
import os
import sys
import copy


DEFAULT_CONFIG: dict = {
    "device": {
        "screen_width": 1080,
        "screen_height": 2408,
        "brush_slider_x": 42,
        "palette_y": 2200,
        "safe_x_boundary": 120,
        "palette_x_positions": [432, 515, 597, 680, 763, 845, 928],
    },
    "brush_config": {
        "1": {"y": 1511, "width": 27},
        "2": {"y": 1350, "width": 42},
        "3": {"y": 1176, "width": 69},
        "4": {"y": 1022, "width": 96},
        "5": {"y": 869, "width": 114},
    },
    "drawing": {
        "fill_step_size": 6,
        "contour_epsilon": 0.0012,
        "swipe_duration_long": 800,
        "swipe_duration_mid": 300,
        "swipe_duration_short": 120,
        "inter_swipe_delay": 0.15,
        "post_background_settle": 1.5,
        "pre_draw_countdown": 3,
    },
    "palette_page_swipe": {
        "start_x": 850,
        "end_x": 450,
        "duration": 400,
        "settle_delay": 1.5,
    },
    "spectrum": {},
}

CONFIG_FILE: str = "config.json"


def get_config_path() -> str:
    """Return the absolute path to config.json next to the running script or exe.

    Handles both PyInstaller-frozen executables (where ``sys._MEIPASS`` is set)
    and normal Python script execution.

    Returns:
        str: Absolute path to the configuration file.
    """
    if getattr(sys, "frozen", False):
        # Running as a PyInstaller bundle — resolve relative to the exe.
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as a normal .py script.
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, CONFIG_FILE)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict.

    Keys present in *base* but missing from *overrides* are preserved so that
    newly-added default keys automatically appear in older config files.

    Args:
        base: The base dictionary (typically ``DEFAULT_CONFIG``).
        overrides: User-supplied overrides loaded from disk.

    Returns:
        dict: A new dictionary containing the merged result.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: str = None) -> dict:
    """Load configuration from a JSON file, merged with defaults.

    If the file does not exist a warning is printed recommending calibration,
    and a copy of ``DEFAULT_CONFIG`` is returned instead.

    Args:
        config_path: Optional explicit path to the JSON config file.
                     When *None*, ``get_config_path()`` is used.

    Returns:
        dict: The fully-merged configuration dictionary.

    Raises:
        json.JSONDecodeError: If the config file contains invalid JSON.
    """
    if config_path is None:
        config_path = get_config_path()

    if not os.path.exists(config_path):
        print(
            f"[WARNING] Config file not found at '{config_path}'. "
            "Using default configuration. Calibration is recommended."
        )
        return copy.deepcopy(DEFAULT_CONFIG)

    with open(config_path, "r", encoding="utf-8") as fh:
        user_config: dict = json.load(fh)

    merged = _deep_merge(DEFAULT_CONFIG, user_config)
    return merged


def save_config(config: dict, config_path: str = None) -> None:
    """Save a configuration dictionary to a JSON file.

    Args:
        config: The configuration dictionary to persist.
        config_path: Optional explicit path for the output file.
                     When *None*, ``get_config_path()`` is used.
    """
    if config_path is None:
        config_path = get_config_path()

    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)

    print(f"[INFO] Configuration saved to '{config_path}'.")
