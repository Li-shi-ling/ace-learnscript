#!/usr/bin/env python3
"""One-click training entrypoint driven by a single YAML config file."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None

from eval.roleplay.data_processor import DataProcessor, load_data


def load_yaml(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def resolve_config_value(value: Any) -> str:
    """Resolve raw value or ENV indirection such as ${ENV:OPENAI_API_KEY}."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("${ENV:") and text.endswith("}"):
        env_name = text[6:-1].strip()
        return os.getenv(env_name, "").strip()
    return text


def apply_llm_runtime_env(llm: Dict[str, Any]) -> str:
    """Apply LLM endpoint/key from config into runtime env for ACE client initialization."""
    api_provider = llm.get("api_provider", "openai_compatible")
    api_base = resolve_config_value(llm.get("api_base"))
    api_key = resolve_config_value(llm.get("api_key"))

    if api_base:
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base
        if api_provider == "openai":
            # If custom base is provided, switch to compatible mode automatically.
            api_provider = "openai_compatible"

    if api_key:
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    return api_provider


def preprocess_roleplay_data(config: Dict[str, Any], data_processor: DataProcessor):
    mode = config["experiment"]["mode"]
    dataset = config["data"]

    if mode in ["online", "eval_only"]:
        test_samples = data_processor.process_task_data(load_data(dataset["test_data"]))
        return None, None, test_samples

    has_explicit_split = ("train_data" in dataset) or ("val_data" in dataset)
    has_source_data = "source_data" in dataset

    if has_explicit_split and has_source_data:
        raise ValueError("data cannot define both explicit train/val and source_data")

    if has_explicit_split:
        if "train_data" not in dataset or "val_data" not in dataset:
            raise ValueError("train_data and val_data must appear together")
        train_samples = data_processor.process_task_data(load_data(dataset["train_data"]))
        val_samples = data_processor.process_task_data(load_data(dataset["val_data"]))
        test_samples = data_processor.process_task_data(load_data(dataset["test_data"])) if "test_data" in dataset else []
        return train_samples, val_samples, test_samples

    if has_source_data:
        rows = load_data(dataset["source_data"])
        aliases = DataProcessor.load_role_aliases(dataset.get("role_aliases_config"))
        all_samples = DataProcessor.create_roleplay_samples_from_dialog_csv(
            rows,
            focus_role=dataset.get("focus_role", "hiro"),
            context_turn_window=dataset.get("context_turn_window", 6),
            min_context_chars=dataset.get("min_context_chars", 1),
            role_aliases=aliases,
        )
        return DataProcessor.split_samples(
            all_samples,
            train_ratio=dataset.get("train_ratio", 0.8),
            val_ratio=dataset.get("val_ratio", 0.1),
            seed=dataset.get("split_seed", 42),
        )

    raise ValueError("data section must provide either train/val split or source_data")


def read_initial_playbook(path: str | None) -> str | None:
    if not path:
        return None
    playbook_path = Path(path)
    if not playbook_path.exists():
        raise FileNotFoundError(f"initial_playbook_path not found: {path}")
    return playbook_path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run training from a single YAML config")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)

    experiment = config.get("experiment", {})
    llm = config.get("llm", {})
    training = config.get("training", {})
    output = config.get("output", {})

    task_name = experiment.get("task_name", "roleplay")
    mode = experiment.get("mode", "offline")
    api_provider = apply_llm_runtime_env(llm)

    data_processor = DataProcessor(
        task_name=task_name,
        critic_model=llm.get("critic_model", "gpt-4o"),
        pass_threshold=training.get("style_pass_threshold", 8),
        critic_api_key=resolve_config_value(llm.get("critic_api_key")) or resolve_config_value(llm.get("api_key")),
        critic_api_base=resolve_config_value(llm.get("critic_api_base")) or resolve_config_value(llm.get("api_base")),
    )

    train_samples, val_samples, test_samples = preprocess_roleplay_data(config, data_processor)

    from ace import ACE

    ace_system = ACE(
        api_provider=api_provider,
        generator_model=llm.get("generator_model", "gpt-4o-mini"),
        reflector_model=llm.get("reflector_model", "gpt-4o-mini"),
        curator_model=llm.get("curator_model", "gpt-4o-mini"),
        max_tokens=llm.get("max_tokens", 4096),
        initial_playbook=read_initial_playbook(output.get("initial_playbook_path")),
        playbook_template=experiment.get("playbook_template", "style"),
    )

    run_config = {
        "num_epochs": training.get("num_epochs", 1),
        "max_num_rounds": training.get("max_num_rounds", 3),
        "curator_frequency": training.get("curator_frequency", 1),
        "eval_steps": training.get("eval_steps", 100),
        "online_eval_frequency": training.get("online_eval_frequency", 15),
        "save_steps": training.get("save_steps", 50),
        "playbook_token_budget": training.get("playbook_token_budget", 80000),
        "task_name": task_name,
        "mode": mode,
        "json_mode": llm.get("json_mode", False),
        "no_ground_truth": training.get("no_ground_truth", False),
        "save_dir": output.get("save_dir", "./results"),
        "test_workers": training.get("test_workers", 20),
        "training_objective": experiment.get("training_objective", ""),
    }

    metadata_path = Path(run_config["save_dir"]) / "last_run_request.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    results = ace_system.run(
        mode=mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=run_config,
    )
    print(f"Final results: {results}")


if __name__ == "__main__":
    main()
