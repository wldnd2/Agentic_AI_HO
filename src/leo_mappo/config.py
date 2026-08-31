from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path,
    override_path: str | Path | Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if override_path is None:
        override_paths: list[str | Path] = []
    elif isinstance(override_path, (str, Path)):
        override_paths = [override_path]
    else:
        override_paths = list(override_path)
    for current_override in override_paths:
        with Path(current_override).open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        config = _deep_update(deepcopy(config), override)
    return config
