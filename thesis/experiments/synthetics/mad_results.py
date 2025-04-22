#!/usr/bin/env python3
import os
import glob
import pandas as pd


def parse_settings_from_dir(log_dir):
    """
    Split the directory basename on '_' and parse every key-value pair of the form key-value.
    Replaces '#' with '.' in values, and converts to int or float when possible.
    Also renames short keys to more descriptive names.
    """
    rename_map = {
        't':        'task_code',
        'vs':       'vocab_size',
        'sl':       'seq_len',
        'ntr':      'num_train',
        'nte':      'num_eval',
        'km':       'km',
        'vm':       'vm',
        'mq':       'mq',
        'fn':       'fn',
        'nvs':      'nvs',
        'ntc':      'ntc',
        'bs':       'batch_size',
        'e':        'epochs',
        'lr':       'lr',
        'wd':       'wd',
        'opt':      'optimizer',
        'sch':      'scheduler',
        's':        'seed',
        'model':    'model_id'
    }

    base = os.path.basename(log_dir)
    settings = {}
    for part in base.split('_'):
        if '-' not in part:
            continue
        key, raw_val = part.split('-', 1)
        # normalize decimal separator
        val = raw_val.replace('#', '.')
        # convert to int/float if possible
        try:
            val_parsed = int(val)
        except ValueError:
            try:
                val_parsed = float(val)
            except ValueError:
                val_parsed = val
        # rename key if in map
        nice_key = rename_map.get(key, key)
        settings[nice_key] = val_parsed

    return settings


def calculate_mad_score(logs_base_dir, target_task_code):
    """
    Implements the MAD scoring protocol for a single task code.
    """
    # 1) find every results.csv under any 't-{task}_*' folder
    pattern = os.path.join(logs_base_dir, '**', f't-{target_task_code}_*', 'results.csv')
    result_files = glob.glob(pattern, recursive=True)
    if not result_files:
        raise FileNotFoundError(f"No results.csv found for task '{target_task_code}' under {logs_base_dir}")

    records = []
    for file_path in result_files:
        log_dir = os.path.dirname(file_path)
        settings = parse_settings_from_dir(log_dir)

        # ensure this really is the right task
        if settings.get('task_code') != target_task_code:
            continue

        # load the single-line CSV
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        if df.empty or 'test_acc' not in df.columns:
            continue

        # grab the *final* test_acc (in case CSV has multiple epochs)
        test_acc = float(df['test_acc'].iloc[-1])
        settings['test_acc'] = test_acc
        records.append(settings)

    if not records:
        raise RuntimeError("No valid runs found or parsed.")

    df = pd.DataFrame(records)

    # 2) identify which columns are *metrics* or *hyperparams* and drop them from grouping
    metric_cols = {'train_acc','train_ppl','train_loss','test_acc','test_ppl','test_loss'}
    hyperparam_cols = {'lr','wd','optimizer','scheduler','seed','model_id','epochs','batch_size'}
    drop_cols = metric_cols.union(hyperparam_cols).union({'task_code'})

    difficulty_cols = [c for c in df.columns if c not in drop_cols]
    if not difficulty_cols:
        raise RuntimeError("No difficulty settings detected—check your parsing logic.")

    # 3) for each unique difficulty combination, pick the run with max test_acc
    idx = df.groupby(difficulty_cols)['test_acc'].idxmax()
    best_df = df.loc[idx].reset_index(drop=True)

    # 4) final MAD score = mean of those best test_acc values
    final_score = best_df['test_acc'].mean()
    print(f"\nGrouped by difficulty settings: {difficulty_cols}")
    print(f"Found {len(best_df)} unique difficulty settings:")
    print(best_df[difficulty_cols + ['lr','wd','test_acc']].to_string(index=False))
    print(f"\n→ Final MAD Score for {target_task_code}: {final_score:.6f}\n")

    return final_score


if __name__ == "__main__":
    # ====== USER CONFIGURATION ======
    LOGS_BASE_DIRECTORY = "/scratch/gpfs/mn4560/thesis/thesis/experiments/synthetics/mad-lab/benchmark/logs"
    TASK_CODE_TO_SCORE   = "NR"   # e.g. 'CR', 'FCR', 'NR', 'SC', 'C', 'M'
    # ==================================

    try:
        calculate_mad_score(LOGS_BASE_DIRECTORY, TASK_CODE_TO_SCORE)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
