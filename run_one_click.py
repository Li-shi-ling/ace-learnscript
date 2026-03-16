#!/usr/bin/env python3
"""One-click training entrypoint driven by a single YAML config file."""

import argparse
import json
import os
from datetime import datetime
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


def describe_split_counts(all_samples_count: int, train_ratio: float, val_ratio: float) -> None:
    train_n = int(all_samples_count * train_ratio)
    val_n = int(all_samples_count * val_ratio)
    test_n = all_samples_count - train_n - val_n
    print(
        f"[数据切分说明] 总样本={all_samples_count}，train_ratio={train_ratio}，val_ratio={val_ratio}，"
        f"因此 train={train_n}, val={val_n}, test={test_n}。"
    )


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
        train_ratio = dataset.get("train_ratio", 0.8)
        val_ratio = dataset.get("val_ratio", 0.1)
        describe_split_counts(len(all_samples), train_ratio, val_ratio)
        train_samples, val_samples, test_samples = DataProcessor.split_samples(
            all_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=dataset.get("split_seed", 42),
        )
        max_train_samples = int(dataset.get("max_train_samples", 0) or 0)
        max_val_samples = int(dataset.get("max_val_samples", 0) or 0)
        max_test_samples = int(dataset.get("max_test_samples", 0) or 0)
        if max_train_samples > 0:
            train_samples = train_samples[:max_train_samples]
        if max_val_samples > 0:
            val_samples = val_samples[:max_val_samples]
        if max_test_samples > 0:
            test_samples = test_samples[:max_test_samples]
        return train_samples, val_samples, test_samples

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
    print(f"[启动] 已加载配置文件: {args.config}")

    experiment = config.get("experiment", {})
    llm = config.get("llm", {})
    training = config.get("training", {})
    output = config.get("output", {})

    task_name = experiment.get("task_name", "roleplay")
    mode = experiment.get("mode", "offline")
    api_provider = apply_llm_runtime_env(llm)
    print(
        f"[LLM] provider={api_provider}, generator={llm.get('generator_model')}, "
        f"reflector={llm.get('reflector_model')}, curator={llm.get('curator_model')}"
    )

    data_processor = DataProcessor(
        task_name=task_name,
        critic_model=llm.get("critic_model", "gpt-4o"),
        pass_threshold=training.get("style_pass_threshold", 8),
        critic_api_key=resolve_config_value(llm.get("critic_api_key")) or resolve_config_value(llm.get("api_key")),
        critic_api_base=resolve_config_value(llm.get("critic_api_base")) or resolve_config_value(llm.get("api_base")),
    )

    train_samples, val_samples, test_samples = preprocess_roleplay_data(config, data_processor)
    print(f"[数据] train={0 if train_samples is None else len(train_samples)}, val={0 if val_samples is None else len(val_samples)}, test={0 if test_samples is None else len(test_samples)}")

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

    save_dir = output.get("save_dir", "./results")
    resume_enabled = bool(training.get("resume", False))
    run_name = output.get("run_name")
    resume_run_path = output.get("resume_run_path")

    # Ensure stable output path for resume: if resume=true and path is omitted,
    # fall back to save_dir/run_name, then save_dir/last_run_path.txt.
    save_dir_path = Path(save_dir)
    last_run_path_file = save_dir_path / "last_run_path.txt"
    if resume_enabled and not resume_run_path:
        if run_name:
            resume_run_path = str(save_dir_path / run_name)
        elif last_run_path_file.exists():
            resume_run_path = last_run_path_file.read_text(encoding="utf-8").strip()

    # If user did not set run_name, generate one in runner so output path is predictable.
    if not run_name and not resume_run_path:
        run_name = f"ace_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_name}_{mode}"

    if resume_enabled and not resume_run_path:
        raise ValueError(
            "resume=true 但未找到可恢复目录。请设置 output.resume_run_path 或 output.run_name。"
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
        "save_dir": save_dir,
        "test_workers": training.get("test_workers", 20),
        "training_objective": experiment.get("training_objective", ""),
        "resume": resume_enabled,
        "resume_run_path": resume_run_path,
        "run_name": run_name,
        "context_max_chars": training.get("context_max_chars", 1600),
        "question_max_chars": training.get("question_max_chars", 800),
        "enable_post_train_generation": training.get("enable_post_train_generation", True),
        "log_language": training.get("log_language", "zh"),
        "max_train_samples": config.get("data", {}).get("max_train_samples", 0),
        "max_val_samples": config.get("data", {}).get("max_val_samples", 0),
        "max_test_samples": config.get("data", {}).get("max_test_samples", 0),
    }

    metadata_path = Path(run_config["save_dir"]) / "last_run_request.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    effective_run_path = (
        Path(run_config["resume_run_path"]) if run_config["resume_run_path"]
        else Path(run_config["save_dir"]) / str(run_config["run_name"])
    )
    last_run_path_file.write_text(str(effective_run_path), encoding="utf-8")

    print("[训练] 开始运行 ACE...（支持中断后续训）")
    if run_config["resume"]:
        print(f"[续训] 已开启 resume，恢复目录={run_config['resume_run_path']}")
    else:
        print(f"[输出] 本次运行目录: {effective_run_path}")
    results = ace_system.run(
        mode=mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=run_config,
    )
    print(f"[完成] 最终结果: {results}")


if __name__ == "__main__":
    main()
