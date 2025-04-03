import time
import sys

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from transformers import AutoModelForCausalLM, AutoTokenizer, MambaForCausalLM

torch.set_float32_matmul_precision("high")


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


def load_model(checkpoint, model_class, tokenizer_class, device, dtype):
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(checkpoint, device_map=device, torch_dtype=dtype).to(device)
    # Add throttling after model loading to let GPU cool down
    time.sleep(0.1)
    return model, tokenizer


def warmup_model(model, inputs, device, iterations=3):
    for _ in range(iterations):
        _ = model.generate(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        # Add throttling between warmup runs
        time.sleep(0.1)


def stream_generate(model, tokenizer, inputs, max_new_tokens=150, temperature=0.7, top_k=50):
    model.eval()
    input_ids = inputs["input_ids"].to(model.device)  # [1, seq_len]
    attention_mask = inputs.get("attention_mask").to(model.device)  # [1, seq_len]

    generated_ids = input_ids.clone()
    start_time = time.time()
    new_tokens = 0

    print("Streaming output:", end=" ")
    sys.stdout.flush()

    sample_rng = torch.Generator(device=model.device)
    sample_rng.manual_seed(1746)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            if temperature > 0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)  # [1, vocab_size]

            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)  # [1, top_k]
            ix = torch.multinomial(top_k_probs, 1, generator=sample_rng)  # [1, 1]
            next_token = torch.gather(top_k_indices, -1, ix)  # [1, 1]

            # Ensure next_token is 2D before concatenation
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)  # [1, seq_len + 1]
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)  # [1, seq_len + 1]

            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)  # Decode single token
            print(token_str, end="", flush=True)

            new_tokens += 1

            # Add power throttling every 10 tokens during generation
            if new_tokens % 10 == 0:
                time.sleep(0.1)

            if next_token[0].item() == tokenizer.eos_token_id:
                break

    elapsed_time = time.time() - start_time
    tokens_per_sec = new_tokens / elapsed_time if elapsed_time > 0 else 0

    print()  # Newline after streaming
    return generated_ids, tokens_per_sec


def benchmark_model(model, inputs, min_run_time=0.2):
    timer = benchmark.Timer(
        stmt="model.generate(**inputs); time.sleep(0.1)",  # Add throttling after each benchmark run
        globals={"model": model, "inputs": inputs, "time": time},
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    return measurement.median


def main():
    device, dtype = setup_device()

    # Model 1: SmolLM-360M
    model1, tokenizer1 = load_model("HuggingFaceTB/SmolLM-360M", AutoModelForCausalLM, AutoTokenizer, device, dtype)
    inputs1 = tokenizer1("Hi, I'm a student at Princeton University, and", return_tensors="pt", padding=True)
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}

    # Throttle between model loads
    time.sleep(0.1)

    # Model 2: Mamba-370m-hf
    model2, tokenizer2 = load_model("state-spaces/mamba-370m-hf", MambaForCausalLM, AutoTokenizer, device, dtype)
    inputs2 = tokenizer2("Hi, I'm a student at Princeton University, and", return_tensors="pt", padding=True)
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}

    # Streaming generation with tokens/sec
    print("\nGenerating text samples...")
    print("\nSmolLM-360M:")
    _, tokens_sec1 = stream_generate(model1, tokenizer1, inputs1)

    # Throttle between models
    time.sleep(0.1)

    print("\nMamba-370m-hf:")
    _, tokens_sec2 = stream_generate(model2, tokenizer2, inputs2)

    # Memory footprint
    if hasattr(model1, "get_memory_footprint"):
        mem1 = model1.get_memory_footprint() / 1e6
        mem2 = model2.get_memory_footprint() / 1e6

    # Benchmarking
    print("\nRunning benchmarks...")

    print("Warming up model 1...")
    warmup_model(model1, inputs1, device)
    time.sleep(0.1)  # Throttle after warmup
    time1 = benchmark_model(model1, inputs1)

    # Throttle between benchmarks
    time.sleep(0.1)

    print("Warming up model 2...")
    warmup_model(model2, inputs2, device)
    time.sleep(0.1)  # Throttle after warmup
    time2 = benchmark_model(model2, inputs2)

    # Present results in a clean tabular format
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(f"{'Metric':<20} {'SmolLM-360M':<15} {'Mamba-370m-hf':<15}")
    print("-" * 50)
    print(f"{'Tokens/sec':<20} {tokens_sec1:<15.2f} {tokens_sec2:<15.2f}")
    print(f"{'Memory (MB)':<20} {mem1:<15.2f} {mem2:<15.2f}")
    print(f"{'Inference speed (s)':<20} {time1:<15.4f} {time2:<15.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
