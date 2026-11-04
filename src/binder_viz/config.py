from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re

import yaml


@dataclass
class Folder:
    path: str
    target: str = ""
    model: str = ""

    def __post_init__(self):
        self.path = str(Path(self.path).expanduser().resolve())
        name = Path(self.path).name
        self.target = self.target.strip() or name
        self.model = self.model.strip() or infer_model(name)
        if "__" in self.model:
            raise ValueError("Model names cannot contain '__'.")


def infer_model(name: str) -> str:
    text = name.lower()
    for pattern, model in [(r"opendde", "opendde"), (r"esmfold2|esmf2", "esmfold2"),
                           (r"af2|alphafold", "af2"), (r"boltz", "boltz")]:
        if re.search(pattern, text):
            return model
    return "model"


def infer_target(name: str, experimental_targets: list[str]) -> str:
    ignored = {"af2", "alphafold", "alphafold2", "opendde", "esmfold2", "esmf2", "boltz", "boltz2",
               "binder", "binders", "eval", "evaluation", "evaluations", "result", "results",
               "prediction", "predictions", "run", "runs", "batch", "traj", "stats"}
    tokens = [token for token in re.findall(r"[a-z0-9]+", name.casefold())
              if token not in ignored and not re.fullmatch(r"(?:batch|run|seed|replicate)\d+", token)]
    candidates = {"".join(tokens[start:end]) for start in range(len(tokens)) for end in range(start + 1, len(tokens) + 1)}
    normalized = {target: "".join(re.findall(r"[a-z0-9]+", target.casefold())) for target in experimental_targets}
    exact = [target for target, value in normalized.items() if value and value in candidates]
    if exact:
        return exact[0] if len(exact) == 1 else name
    scores = []
    for target, value in normalized.items():
        if len(value) < 6:
            continue
        score = max((SequenceMatcher(None, value, candidate, autojunk=False).ratio() for candidate in candidates
                     if re.findall(r"\d+", value) == re.findall(r"\d+", candidate)), default=0.0)
        scores.append((score, target))
    scores.sort(reverse=True)
    if scores and scores[0][0] >= 0.88 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.08):
        return scores[0][1]
    return name


@dataclass
class Aggregation:
    mode: str = "best"
    selector: str = "iptm"
    direction: str = "higher_better"
    overrides: dict[str, str] = field(default_factory=dict)
    std: bool = True
    composite: dict = field(default_factory=dict)

    def __post_init__(self):
        modes = {"none", "best", "worst", "mean", "median"}
        if self.mode not in modes or any(value not in modes - {"none"} for value in self.overrides.values()):
            raise ValueError("Unsupported aggregation mode or override.")
        if self.direction not in {"higher_better", "lower_better"}:
            raise ValueError("Selector direction must be higher_better or lower_better.")


@dataclass
class SessionConfig:
    folders: list[Folder] = field(default_factory=list)
    phase2: Aggregation = field(default_factory=Aggregation)
    phase3: Aggregation = field(default_factory=lambda: Aggregation(selector="total_score", direction="lower_better"))
    include_phase3: bool = True
    duplicates: str = "first"
    registry_path: str = ""
    registry: list[dict] = field(default_factory=list)
    experiments: list[dict] = field(default_factory=list)
    label_mapping: dict[str, str] = field(default_factory=dict)
    experimental_duplicates: str = "flag"
    filters: dict = field(default_factory=dict)
    composites: list[dict] = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    bootstrap_samples: int = 200
    random_seed: int = 42

    def __post_init__(self):
        if self.duplicates not in {"first", "last", "error"}:
            raise ValueError("Duplicate policy must be first, last or error.")
        if self.experimental_duplicates not in {"flag", "aggregate", "error"}:
            raise ValueError("Unsupported experimental duplicate policy.")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> SessionConfig:
        values = dict(data or {})
        values["folders"] = [Folder(**row) for row in values.get("folders", [])]
        for stage in ("phase2", "phase3"):
            if stage in values:
                values[stage] = Aggregation(**values[stage])
        return cls(**values)

    @classmethod
    def from_yaml(cls, text: str) -> SessionConfig:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Session YAML must contain a mapping.")
        return cls.from_dict(data)