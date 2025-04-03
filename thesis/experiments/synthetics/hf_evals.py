import os
import time

from typing import Dict, List, Optional

import lm_eval as evaluator
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, MambaForCausalLM

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision("high")


class ModelConfig:
    def __init__(
        self,
        name: str,
        checkpoint: str,
        model_class,
        tokenizer_class,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        self.name = name
        self.checkpoint = checkpoint
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.dtype = dtype
        self.device = device


def setup_device() -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


def evaluate_model(
    model_config: ModelConfig,
    tasks: List[str],
    tasks_fewshot: Dict[str, int],
    batch_size: str = "auto",
    cache_dir: Optional[str] = None,
) -> Dict:
    """Evaluate a model on specified tasks."""
    print(f"\nEvaluating {model_config.name}...")
    start_time = time.time()

    # Convert dtype to string format
    dtype_str = "bfloat16" if model_config.dtype == torch.bfloat16 else "float32"

    # Prepare model arguments
    model_args = (
        f"pretrained={model_config.checkpoint},"
        "trust_remote_code=True,"
        f"dtype={dtype_str},"
        f"device_map={model_config.device}"
    )
    if cache_dir:
        model_args += f",cache_dir={cache_dir}"

    # Run evaluation for each task
    all_results = {}
    for task in tasks:
        print(f"\nRunning task: {task}")
        eval_kwargs = dict(
            model="hf", model_args=model_args, tasks=[task], batch_size=batch_size, device=model_config.device
        )

        # Add few-shot setting if specified
        few_shot_value = tasks_fewshot.get(task, -1)
        if few_shot_value != -1:
            eval_kwargs["num_fewshot"] = few_shot_value

        results = evaluator.simple_evaluate(**eval_kwargs)
        task_result = results["results"].get(task, {})
        all_results[task] = task_result

    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    return all_results


def format_results(model_results: Dict[str, Dict]) -> None:
    """Print results in a clean tabular format."""

    # Get all unique tasks and metrics
    all_tasks = set()
    all_metrics = set()
    for results in model_results.values():
        all_tasks.update(results.keys())
        for task_results in results.values():
            all_metrics.update(task_results.keys())

    # Print header
    print("\n" + "=" * 120)
    print("EVALUATION RESULTS")
    print("=" * 120)

    # Organize by model first, then by task for clearer hierarchy
    for model_name in sorted(model_results.keys()):
        print(f"\nModel: {model_name}")
        print("-" * 80)

        # Print header row with tasks
        header = "Metric".ljust(25)
        for task in sorted(all_tasks):
            header += f"{task}".rjust(15)
        print(header)
        print("-" * 80)

        # Print metrics for this model
        for metric in sorted(all_metrics):
            row = metric.ljust(25)
            for task in sorted(all_tasks):
                if task in model_results[model_name] and metric in model_results[model_name][task]:
                    value = model_results[model_name][task][metric]
                    # Handle both string and numeric values
                    if isinstance(value, (int, float)):
                        row += f"{value:15.4f}"
                    else:
                        row += f"{str(value):>15}"
                else:
                    row += " " * 15  # Add padding for missing values
            print(row)

        print("-" * 80)


def main():
    # Define tasks and their few-shot settings
    tasks = [
        "hellaswag",
        # "piqa",
        # "siqa",
        # "boolq",
        # "winogrande",
        # "commonsense_qa",
        # "openbookqa",
        # "arc",
        # "arc_easy",
        # "arc_challenge",
    ]

    tasks_fewshot = {
        "hellaswag": None,
        # "piqa": 0,
        # "siqa": 0,
        # "boolq": 0,
        # "winogrande": -1,
        # "commonsense_qa": 7,
        # "openbookqa": -1,
        # "arc": -1,
        # "arc_easy": -1,
        # "arc_challenge": -1,
    }

    # Setup device and models
    device, dtype = setup_device()

    model_configs = [
        ModelConfig(
            name="SmolLM-360M",
            checkpoint="HuggingFaceTB/SmolLM-360M",
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
            dtype=dtype,
            device=device,
        ),
        ModelConfig(
            name="Mamba-370m-hf",
            checkpoint="state-spaces/mamba-370m-hf",
            model_class=MambaForCausalLM,
            tokenizer_class=AutoTokenizer,
            dtype=dtype,
            device=device,
        ),
    ]

    # Run evaluations
    all_model_results = {}
    for config in model_configs:
        results = evaluate_model(
            config,
            tasks,
            tasks_fewshot,
            batch_size="auto",

            # Change this to your own cache folder lol
            cache_dir="/scratch/gpfs/mn4560/hazan-lab/tensorized_filters/tensorized_filters/eval/cache",
        )
        all_model_results[config.name] = results

    # Print results in a clean format
    format_results(all_model_results)


if __name__ == "__main__":
    main()
