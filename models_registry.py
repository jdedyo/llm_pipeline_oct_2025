from pathlib import Path
from typing import Dict, Any, Iterable
from transformers import TrainingArguments
from copy import deepcopy

class ModelRegistry:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def add(self, name: str, *, type: str, model_id: str,
            save_path: Path, base_model_path: Path, 
            train_data_path: Path, prompt_path: Path, 
            train_epochs: int, train_ans_col: str,
            training_args: TrainingArguments) -> None:
            
        self._data[name] = {
            "name": name,
            "type": type,
            "model_id": model_id,
            "save_path": save_path,
            "base_model_path": base_model_path,
            "train_data_path": train_data_path,
            "prompt_path": prompt_path,
            "train_epochs": train_epochs,
            "train_ans_col": train_ans_col,
            "training_args": deepcopy(training_args)
        }

    def choices(self) -> Iterable[str]:
        return tuple(self._data.keys())

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self._data:
            raise ValueError(f"Invalid model '{name}'. Must be one of: {', '.join(self.choices())}.")
        return self._data[name]

    def append_suffix_to_save_paths(self, suffix: str) -> None:
        for cfg in self._data.values():
            p: Path = cfg["save_path"]
            cfg["save_path"] = p.parent / f"{p.name}_{suffix}"