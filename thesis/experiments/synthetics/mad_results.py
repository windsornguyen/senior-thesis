#!/usr/bin/env python3

import os
import glob
import pandas as pd
from typing import Optional, List


def parse_settings_from_dir(log_dir: str) -> dict:
    """
    Parse hyperparameters and experimental settings from the log directory name.
    """
    rename_map = {
        "t": "task_code",
        "vs": "vocab_size",
        "sl": "seq_len",
        "ntr": "num_train",
        "nte": "num_eval",
        "km": "km",
        "vm": "vm",
        "mq": "mq",
        "fn": "fn",
        "nvs": "nvs",
        "ntc": "ntc",
        "bs": "batch_size",
        "e": "epochs",
        "lr": "lr",
        "wd": "wd",
        "opt": "optimizer",
        "sch": "scheduler",
        "s": "seed",
        "model": "model_id",
    }

    base = os.path.basename(log_dir)
    settings = {}
    for part in base.split("_"):
        if "-" not in part:
            continue
        key, raw_val = part.split("-", 1)
        val = raw_val.replace("#", ".")
        try:
            val_parsed = int(val)
        except ValueError:
            try:
                val_parsed = float(val)
            except ValueError:
                val_parsed = val
        settings[rename_map.get(key, key)] = val_parsed

    return settings


def calculate_mad_score(
    logs_base_dir: str,
    target_task_code: str,
    model_name_filter: Optional[str] = None,
    metric: str = "test_acc",
    verbose: bool = True,
) -> float:
    """
    Calculate the MAD score for a given task.

    Args:
        logs_base_dir (str): Base directory containing log subdirectories.
        target_task_code (str): Task code to filter (e.g., 'CR', 'SC', 'M').
        model_name_filter (Optional[str]): Only evaluate models whose name contains this substring.
        metric (str): Metric to maximize ('test_acc' or others).
        verbose (bool): If True, print per-difficulty best scores.

    Returns:
        float: Final MAD score for the specified task.
    """
    pattern = os.path.join(logs_base_dir, "**", f"t-{target_task_code}_*", "results.csv")
    result_files = glob.glob(pattern, recursive=True)
    if not result_files:
        raise FileNotFoundError(f"No results.csv found for task '{target_task_code}' under {logs_base_dir}")

    records = []
    for file_path in result_files:
        log_dir = os.path.dirname(file_path)
        settings = parse_settings_from_dir(log_dir)

        if settings.get("task_code") != target_task_code:
            continue
        if model_name_filter and model_name_filter not in settings.get("model_id", ""):
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        if df.empty or metric not in df.columns:
            continue

        final_val = float(df[metric].iloc[-1])
        settings[metric] = final_val
        records.append(settings)

    if not records:
        raise RuntimeError("No valid runs found for scoring.")

    df = pd.DataFrame(records)

    metric_cols = {"train_acc", "train_ppl", "train_loss", "test_acc", "test_ppl", "test_loss"}
    hyperparam_cols = {"lr", "wd", "optimizer", "scheduler", "seed", "model_id", "epochs", "batch_size"}
    drop_cols = metric_cols.union(hyperparam_cols).union({"task_code"})

    difficulty_cols = [c for c in df.columns if c not in drop_cols]
    if not difficulty_cols:
        raise RuntimeError("No difficulty settings detected.")

    idx = df.groupby(difficulty_cols)[metric].idxmax()
    best_df = df.loc[idx].reset_index(drop=True)

    final_score = best_df[metric].mean()

    if verbose:
        print(f"\nTask: {target_task_code}")
        print(f"Grouped by difficulty settings: {difficulty_cols}")
        print(f"Found {len(best_df)} unique difficulty settings:")
        print(best_df[difficulty_cols + ["lr", "wd", metric]].to_string(index=False))
        print(f"\n→ Final MAD Score for {target_task_code}: {final_score:.6f}\n")

    return final_score


def calculate_mad_score_across_tasks(
    logs_base_dir: str,
    task_codes: List[str],
    model_name_filter: Optional[str] = None,
    metric: str = "test_acc",
    verbose: bool = True,
) -> float:
    """
    Calculate the average MAD score across multiple tasks.

    Args:
        logs_base_dir (str): Base directory containing log subdirectories.
        task_codes (List[str]): List of task codes to evaluate.
        model_name_filter (Optional[str]): Filter models by name substring.
        metric (str): Metric to maximize.
        verbose (bool): Print detailed output per task.

    Returns:
        float: Final averaged MAD score.
    """
    task_scores = []

    for task_code in task_codes:
        try:
            score = calculate_mad_score(
                logs_base_dir, target_task_code=task_code, model_name_filter=model_name_filter, metric=metric, verbose=verbose
            )
            task_scores.append(score)
        except Exception as e:
            print(f"Warning: Skipping task {task_code} due to error: {e}")

    if not task_scores:
        raise RuntimeError("No task scores could be computed.")

    final_mad = sum(task_scores) / len(task_scores)
    print(f"\n==== Summary ====")
    print(f"Average MAD Score across {len(task_scores)} tasks: {final_mad:.6f}\n")
    return final_mad


if __name__ == "__main__":
    # ====== USER CONFIGURATION ======
    LOGS_BASE_DIRECTORY = "/scratch/gpfs/mn4560/thesis/thesis/experiments/synthetics/mad-lab/benchmark/logs"
    TASK_CODES = ["CR", "FCR", "NR", "SC", "C", "M"]  # List of task codes to compute (default 6 MAD tasks)
    MODEL_NAME_FILTER = "SpAttnFull"  # e.g., "SpAttn", "SpAttnFull", or None for no filtering
    METRIC = "test_acc"  # Which metric to maximize; typically 'test_acc'
    VERBOSE = True  # Whether to print detailed per-task results
    # ==================================

    try:
        calculate_mad_score_across_tasks(
            LOGS_BASE_DIRECTORY,
            task_codes=TASK_CODES,
            model_name_filter=MODEL_NAME_FILTER,
            metric=METRIC,
            verbose=VERBOSE,
        )
    except Exception as e:
        print(f"Fatal Error: {e}")
        exit(1)
