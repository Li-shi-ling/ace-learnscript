#!/usr/bin/env python3
import os
import json
import argparse

from ace import ACE
from .data_processor import DataProcessor, load_data


def parse_args():
    parser = argparse.ArgumentParser(description='ACE System - Roleplay style optimization')
    parser.add_argument("--task_name", type=str, default="roleplay")
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="offline", choices=["offline", "online", "eval_only"])

    parser.add_argument("--api_provider", type=str, default="openai", choices=["sambanova", "together", "openai", "openai_compatible"])
    parser.add_argument("--generator_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--reflector_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--curator_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--critic_model", type=str, default="gpt-4o")
    parser.add_argument("--style_pass_threshold", type=int, default=8)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)

    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--save_path", type=str, required=True)

    return parser.parse_args()


def preprocess_data(config, mode, data_processor):
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        test_samples = data_processor.process_task_data(load_data(config["test_data"]))
        return train_samples, val_samples, test_samples

    has_explicit_split = ("train_data" in config) or ("val_data" in config)
    has_source_data = "source_data" in config

    if has_explicit_split and has_source_data:
        raise ValueError("Roleplay config cannot define both train/val paths and source_data. Please choose one mode.")

    # Offline mode supports either explicit train/val/test paths or single CSV auto-split.
    if has_explicit_split:
        if "train_data" not in config or "val_data" not in config:
            raise ValueError("Both train_data and val_data must be provided together in offline mode.")

        train_samples = data_processor.process_task_data(load_data(config["train_data"]))
        val_samples = data_processor.process_task_data(load_data(config["val_data"]))
        test_samples = data_processor.process_task_data(load_data(config["test_data"])) if "test_data" in config else []
        return train_samples, val_samples, test_samples

    if has_source_data:
        source_rows = load_data(config["source_data"])
        focus_role = config.get("focus_role", "hiro")
        context_turn_window = config.get("context_turn_window", 6)
        min_context_chars = config.get("min_context_chars", 1)
        role_aliases = data_processor.load_role_aliases(config.get("role_aliases_config"))

        all_samples = data_processor.create_roleplay_samples_from_dialog_csv(
            source_rows,
            focus_role=focus_role,
            context_turn_window=context_turn_window,
            min_context_chars=min_context_chars,
            role_aliases=role_aliases,
        )

        train_samples, val_samples, test_samples = data_processor.split_samples(
            all_samples,
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.1),
            seed=config.get("split_seed", 42),
        )
        return train_samples, val_samples, test_samples

    raise ValueError("Roleplay config must provide either train/val data paths or source_data for auto split.")


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def main():
    args = parse_args()

    with open("./eval/roleplay/data/sample_config.json", 'r', encoding='utf-8') as f:
        config_map = json.load(f)

    if args.task_name not in config_map:
        raise ValueError(f"Unknown task: {args.task_name}. Available: {list(config_map.keys())}")

    data_processor = DataProcessor(
        task_name=args.task_name,
        critic_model=args.critic_model,
        pass_threshold=args.style_pass_threshold,
    )

    train_samples, val_samples, test_samples = preprocess_data(config_map[args.task_name], args.mode, data_processor)

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=load_initial_playbook(args.initial_playbook_path),
        playbook_template="style",
    )

    run_config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'online_eval_frequency': args.online_eval_frequency,
        'save_steps': args.save_steps,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
    }

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=run_config,
    )
    print(f"Final results: {results}")


if __name__ == "__main__":
    main()
