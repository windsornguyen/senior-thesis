#!/usr/bin/env python3
import torch
import numpy as np
from thesis.experiments.synthetics.registry import registry, IGNORE_IDX
import pprint

NUM_EXAMPLES_TO_SHOW = 2
BATCH_SIZE = 8
NUM_TRAIN = 100  # For validation, generate a small number of training examples
NUM_TEST = 10    # For validation, generate a small number of test examples
DEFAULT_SEQ_LEN = 32
DEFAULT_VOCAB_SIZE = 16
DEVICE = "cpu"  # Use CPU for simplicity

# --- Get available tasks ---
available_tasks = list(registry.tasks.keys())
print(f"Validating tasks: {available_tasks}\n" + "=" * 30 + "\n")

for task_name in available_tasks:
    print(f"--- Task: {task_name} ---")

    # --- Task-specific configurations (adjust as needed) ---
    task_kwargs = {
        "seq_len": DEFAULT_SEQ_LEN,
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "num_train": NUM_TRAIN,
        "num_test": NUM_TEST,
        "batch_size": BATCH_SIZE,
        "backend": "torch",  # Use torch loaders for easier inspection
        "device": DEVICE,
        "in_memory": True,   # In-memory mode for validation
    }

    # Adjust parameters based on task
    if "recall" in task_name:
        task_kwargs["vocab_size"] = DEFAULT_VOCAB_SIZE + 1  # Account for copy prefix
        task_kwargs["noise_vocab_size"] = 0
        task_kwargs["frac_noise"] = 0.0
        if "noisy" in task_name:
            task_kwargs["vocab_size"] = DEFAULT_VOCAB_SIZE + 1 + 4  # copy + noise tokens
            task_kwargs["noise_vocab_size"] = 4
            task_kwargs["frac_noise"] = 0.2
        if "fuzzy" in task_name:
            task_kwargs["vocab_size"] = DEFAULT_VOCAB_SIZE + 2  # copy + pad
            task_kwargs["k_motif_size"] = 2
            task_kwargs["v_motif_size"] = 2

    if "copying" in task_name:
        task_kwargs["num_tokens_to_copy"] = 8
        task_kwargs["seq_len"] = 32

    if "memorization" in task_name:
        task_kwargs["vocab_size"] = DEFAULT_VOCAB_SIZE + 1  # Account for insert token

    if "compression" in task_name:
        task_kwargs["vocab_size"] = DEFAULT_VOCAB_SIZE + 1  # Account for compression token

    if "induction_heads" in task_name:
        task_kwargs["vocab_size"] = DEFAULT_VOCAB_SIZE + 1  # Account for special token
        task_kwargs.pop("num_train", None)
        task_kwargs.pop("num_test", None)
        task_kwargs["num_examples"] = BATCH_SIZE  # Generate just one batch

    if "mqar" in task_name:
        task_kwargs["vocab_size"] = 128  # Example value
        task_kwargs["num_pairs"] = 8
        task_kwargs["seq_len"] = 64
        task_kwargs.pop("num_train", None)
        task_kwargs.pop("num_test", None)
        task_kwargs["num_examples"] = BATCH_SIZE  # Generate just one batch

    print("Task-specific kwargs:")
    pprint.pprint(
        {k: v for k, v in task_kwargs.items() if k not in ["num_train", "num_test", "backend", "device", "in_memory"]}
    )

    try:
        # Special handling for tasks that generate data directly
        if task_name in ["induction_heads", "mqar"]:
            task_fn = registry.get_task(task_name)
            # Remove args not expected by the direct function
            gen_kwargs = {
                k: v
                for k, v in task_kwargs.items()
                if k not in ["batch_size", "num_train", "num_test", "backend", "device", "in_memory"]
            }
            dataset = task_fn(**gen_kwargs)
            from torch.utils.data import DataLoader

            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
            test_loader = train_loader  # For these tasks, only one dataset is produced
            print(f"Generated {len(dataset)} examples directly.")
        else:
            # Use the standard loader creation method, which creates both train and test datasets.
            train_loader, test_loader = registry.create_data_loaders(task_name=task_name, **task_kwargs)
            print(f"Generated {len(train_loader.dataset)} training examples and {len(test_loader.dataset)} test examples.")

        # --- Inspect a Batch from the Training Loader ---
        print("\n--- Training Batch ---")
        train_inputs, train_targets = next(iter(train_loader))
        print(f"Train Batch Shapes: Inputs: {tuple(train_inputs.shape)}, Targets: {tuple(train_targets.shape)}")
        for i in range(min(NUM_EXAMPLES_TO_SHOW, train_inputs.shape[0])):
            print(f"  Train Example {i + 1}:")
            print(f"    Input:  {train_inputs[i].tolist()}")
            print(f"    Target: {train_targets[i].tolist()}")
            if not torch.all(train_targets[i] == IGNORE_IDX):
                non_ignore = train_targets[i][train_targets[i] != IGNORE_IDX]
                print(f"    Target unique non-ignored: {torch.unique(non_ignore).tolist()}")
            else:
                print(f"    Target all ignored ({IGNORE_IDX})")
        print(f"IGNORE_IDX present in train targets: {torch.any(train_targets == IGNORE_IDX).item()}")

        # --- Inspect a Batch from the Test Loader ---
        print("\n--- Test Batch ---")
        test_inputs, test_targets = next(iter(test_loader))
        print(f"Test Batch Shapes: Inputs: {tuple(test_inputs.shape)}, Targets: {tuple(test_targets.shape)}")
        for i in range(min(NUM_EXAMPLES_TO_SHOW, test_inputs.shape[0])):
            print(f"  Test Example {i + 1}:")
            print(f"    Input:  {test_inputs[i].tolist()}")
            print(f"    Target: {test_targets[i].tolist()}")
            if not torch.all(test_targets[i] == IGNORE_IDX):
                non_ignore = test_targets[i][test_targets[i] != IGNORE_IDX]
                print(f"    Target unique non-ignored: {torch.unique(non_ignore).tolist()}")
            else:
                print(f"    Target all ignored ({IGNORE_IDX})")
        print(f"IGNORE_IDX present in test targets: {torch.any(test_targets == IGNORE_IDX).item()}")

        # --- Optional: Recall Check for in-context recall tasks ---
        if task_name == "in_context_recall" and not task_kwargs.get("multi_query", False):
            print("\n--- Recall Check ---")
            # Check for training data example
            try:
                final_target = train_targets[0, -1].item()
                final_key = train_inputs[0, -2].item()
                train_seq = train_inputs[0].tolist()
                found = False
                for idx in range(len(train_seq) - 3, -1, -2):  # search backward in key-value pairs
                    if train_seq[idx] == final_key:
                        expected_value = train_seq[idx + 1]
                        found = True
                        break
                if found:
                    check = expected_value == final_target
                    print(f"Train Recall Check: Key={final_key}, Expected Value={expected_value}, "
                          f"Target Value={final_target} -> {'PASS' if check else 'FAIL'}")
                else:
                    print(f"Train Recall Check: Could not find previous instance of key {final_key}.")
            except Exception as e:
                print(f"Train Recall Check failed: {e}")

            # Check for test data example
            try:
                final_target_test = test_targets[0, -1].item()
                final_key_test = test_inputs[0, -2].item()
                test_seq = test_inputs[0].tolist()
                found_test = False
                for idx in range(len(test_seq) - 3, -1, -2):
                    if test_seq[idx] == final_key_test:
                        expected_value_test = test_seq[idx + 1]
                        found_test = True
                        break
                if found_test:
                    check_test = expected_value_test == final_target_test
                    print(f"Test Recall Check: Key={final_key_test}, Expected Value={expected_value_test}, "
                          f"Target Value={final_target_test} -> {'PASS' if check_test else 'FAIL'}")
                else:
                    print(f"Test Recall Check: Could not find previous instance of key {final_key_test}.")
            except Exception as e:
                print(f"Test Recall Check failed: {e}")

    except Exception as e:
        print(f"ERROR generating/validating task '{task_name}': {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 30 + "\n")

print("Validation script finished.")
