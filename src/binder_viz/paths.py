from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import yaml


def path_list(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    try:
        parsed = yaml.safe_load(text) if text.startswith(("[", "- ")) else None
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid path list: {error}") from error
    entries = parsed if isinstance(parsed, list) else [entry for row in csv.reader(StringIO(text), skipinitialspace=True) for entry in row]
    if not all(isinstance(entry, str) for entry in entries):
        raise ValueError("Paths must be a list of strings")
    return list(dict.fromkeys(str(Path(entry.strip()).expanduser().resolve()) for entry in entries if entry.strip()))


def input_files(value: str | Path, suffixes: tuple[str, ...]) -> list[Path]:
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"Expected {' / '.join(suffixes)} input: {path}")
        return [path]
    matches = sorted(candidate for candidate in path.rglob("*")
                     if candidate.is_file() and candidate.suffix.lower() in suffixes
                     and not {".git", "__pycache__"}.intersection(candidate.relative_to(path).parts))
    if not matches:
        raise ValueError(f"No {' / '.join(suffixes)} files found in {path}")
    return matches


def input_file(value: str | Path, suffixes: tuple[str, ...]) -> Path:
    matches = input_files(value, suffixes)
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} input files in {value}; choose a specific file")
    return matches[0]


def output_file(value: str | Path, default_name: str) -> Path:
    path = Path(value).expanduser()
    return path / default_name if path.is_dir() else path


def yaml_mapping(value: str, section: str | None = None) -> dict:
    text = value.strip()
    path_like = "\n" not in text and len(text) < 4096 and text not in {"", "{}", "null"}
    if path_like:
        try:
            exists = Path(text).expanduser().exists()
        except OSError:
            exists = False
        if exists or text.startswith(("/", "~/", "./", "../")) or text.lower().endswith((".yaml", ".yml", ".json")):
            text = input_file(text, (".yaml", ".yml", ".json")).read_text()
    data = yaml.safe_load(text) or {}
    if section and isinstance(data, dict) and section in data:
        data = data[section]
    if not isinstance(data, dict):
        raise ValueError("YAML input must contain a mapping")
    return data