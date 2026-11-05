"""Versions of the config grammar.

A config may declare ``schema_version``. Every version accepts the configs of
the versions before it: keys are only ever added. ``epydemix validate
--upgrade`` stamps a config with the current version, in place.

A config without ``schema_version`` is read as the current version. A config
that declares an older version but uses keys introduced later is accepted
with a warning, and one that declares a newer version than this epydemix
understands is rejected.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keys each version introduced, as dotted paths.
INTRODUCED: Dict[int, List[str]] = {
    1: [],  # the original grammar (epydemix 1.5.0)
    2: ["schema_version", "assumptions"],
    3: ["population.data_version"],
    4: ["derived"],
}
CONFIG_SCHEMA_VERSION = max(INTRODUCED)


def _has_path(config: Dict[str, Any], dotted: str) -> bool:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def declared_version(config: Dict[str, Any]) -> Optional[int]:
    version = config.get("schema_version")
    return (
        version if isinstance(version, int) and not isinstance(version, bool) else None
    )


def check(config: Dict[str, Any], errors: list, warnings: list) -> None:
    """Validate ``schema_version`` against the keys the config uses."""
    if "schema_version" not in config:
        return
    version = config["schema_version"]
    if declared_version(config) is None or version < 1:
        errors.append(
            f"schema_version must be a positive integer, got {version!r} "
            f"(current: {CONFIG_SCHEMA_VERSION})"
        )
        return
    if version > CONFIG_SCHEMA_VERSION:
        errors.append(
            f"schema_version {version} is newer than this epydemix understands "
            f"(up to {CONFIG_SCHEMA_VERSION}); upgrade epydemix"
        )
        return
    newer = [
        (key, v)
        for v, keys in INTRODUCED.items()
        if v > version
        for key in keys
        if key != "schema_version" and _has_path(config, key)
    ]
    for key, v in newer:
        warnings.append(
            f"'{key}' was introduced in schema_version {v} but the config declares "
            f"{version}; run `epydemix validate --upgrade` to update it"
        )


def upgrade_file(path: str) -> Dict[str, Any]:
    """Set a config file's ``schema_version`` to the current one, in place.

    YAML files are edited as text, so comments and layout survive: an existing
    top-level ``schema_version:`` line is replaced, otherwise one is added at
    the top. Only the file itself changes, not a ``base:`` it inherits from.

    Returns:
        ``{"path", "from", "to"}``; ``from`` is None for an unversioned file.
    """
    p = Path(path)
    text = p.read_text()
    line = f"schema_version: {CONFIG_SCHEMA_VERSION}"
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        previous = data.get("schema_version")
        data = {"schema_version": CONFIG_SCHEMA_VERSION, **data}
        p.write_text(json.dumps(data, indent=2) + "\n")
        return {"path": str(p), "from": previous, "to": CONFIG_SCHEMA_VERSION}

    pattern = re.compile(r"^schema_version[ \t]*:[ \t]*(\S*).*$", re.MULTILINE)
    match = pattern.search(text)
    if match:
        previous = match.group(1)
        text = text[: match.start()] + line + text[match.end() :]
        try:
            previous = int(previous)
        except ValueError:
            pass
    else:
        previous = None
        text = line + "\n" + text
    p.write_text(text)
    return {"path": str(p), "from": previous, "to": CONFIG_SCHEMA_VERSION}
