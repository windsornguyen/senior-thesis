import time
from dataclasses import dataclass
from typing import List

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from thesis.models.inference import load_model, PackedCausalGeneratorConfig, BasePackedCausalGenerator
from thesis.models.mamba.model import SSM
from thesis.models.tokenizer import Tokenizer
from thesis.models import models_config, model_name_to_cls
from thesis.utils.pretraining_config import JobConfig

def sample_top_p(probs: torch.Tensor, p: float, eps: float = 1e8) -> torch.Tensor:
    probs_sorted, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sorted, dim=-1)
    mask = probs_sum > p
    probs_sorted[mask] = 0.0
    probs_sorted /= (probs_sorted.sum(dim=-1, keepdim=True) + eps)
    next_token = torch.multinomial(probs_sorted, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample_top_k(probs: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    # Sort descending and retain original indices
    probs_sorted, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Mask out everything beyond rank k
    ranks = torch.arange(probs_sorted.size(-1), device=probs.device)
    mask = ranks.unsqueeze(0) >= k  # shape: [batch_size, vocab_size]
    probs_sorted[mask] = 0.0

    # Renormalize
    probs_sorted /= (probs_sorted.sum(dim=-1, keepdim=True) + eps)

    # Sample from the masked distribution
    next_token_sorted = torch.multinomial(probs_sorted, num_samples=1)

    # Map back to the original indices
    next_token = torch.gather(probs_idx, -1, next_token_sorted)
    return next_token

def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Sample next token(s) from logits, optionally applying temperature,
    top-p sampling, or top-k sampling. Falls back to greedy (argmax) if
    temperature == 0.0.
    """
    shape = logits.shape
    
    # Flatten so we have shape [N, vocab_size]
    logits = logits.flatten(end_dim=-2)

    if temperature > 0.0:
        # Temperature scaling, then softmax
        probs = torch.softmax(logits / temperature, dim=-1)
        
        if top_p is not None:
            # Nucleus (top-p) sampling
            next_token = sample_top_p(probs, top_p, eps=eps)
        elif top_k is not None:
            # Top-k sampling
            next_token = sample_top_k(probs, top_k, eps=eps)
        else:
            # Straight multinomial sampling
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        # Greedy (argmax)
        next_token = torch.argmax(logits, dim=-1)

    # Reshape back to original batch shape
    return next_token.view(shape[:-1])

def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = torch.tensor(p, dtype=torch.long)
        l = p.size(0)
        res.append(p)
        lengths.append(l)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths

def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches

class StateCache(nn.Module):
    def __init__(
        self, bsz, n_heads, head_dim, state_dim, conv_size, conv_dim, dtype, device
    ):
        super().__init__()
        state_shape = (bsz, n_heads, head_dim, state_dim)
        if conv_size is None:
            conv_shape = (0,)
        else:
            conv_shape = (bsz, conv_dim, conv_size)

        self.register_buffer(
            "conv_cache",
            torch.zeros(conv_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "state_cache",
            torch.zeros(state_shape, dtype=dtype, device=device),
            persistent=False,
        )

    def reset(self):
        self.conv_cache.zero_()
        self.state_cache.zero_()

@dataclass
class PackedCausalMambaGeneratorConfig(PackedCausalGeneratorConfig):
    pass

class PackedCausalMambaGenerator(BasePackedCausalGenerator):
    def __init__(
        self,
        config: PackedCausalMambaGeneratorConfig,
        model: nn.Module,
        tokenizer: Tokenizer,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.stream = config.stream

        self.max_gen_len = config.max_gen_len
        self.max_tokens = config.max_tokens
        self.max_prompt_len = config.max_prompt_len
        self.until = config.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = config.device

        # Compile if necessary
        self.prefill = torch.compile(self.prefill, disable=not config.compile_prefilling)
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not config.reduce_generation_overhead,
        )

        self.show_progress = config.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[config.dtype]

        self.prefill_tok_id = None
        self.cu_seqlens = None
    
    def clear_cache(self, lengths: torch.Tensor):
        for module in self.model.modules():
            if isinstance(module, SSM):
                module.cache = StateCache(
                    lengths.size(0),
                    module.n_heads,
                    module.head_dim,
                    module.state_dim,
                    module.conv_size,
                    module.conv_dim,
                    self.dtype,
                    self.device,
                )
    
    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        self.clear_cache(lengths)

        self.prefill_tok_id = torch.repeat_interleave(lengths).unsqueeze(0).int()
        self.cu_seqlens = lengths.cumsum(0)
        self.cu_seqlens = torch.cat(
            [torch.tensor([0], device=self.device), self.cu_seqlens]
        ).int()

    @torch.compiler.disable
    def setup_generation(self, lengths):
        pass

    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            cu_seqlens=self.cu_seqlens,
            ssm_impl="ssm",
        )

        return prefill_out

    def generate_next_token(self, current_token):
        out = self.model.forward(
            x=current_token,
            tok_idx=None,
            cu_seqlens=None,
            ssm_impl="ssm_update",
        )
        return out

    def generate(self, prompts):
        # Tokenize
        prompts = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
        ]
        # Truncate
        max_seqlen = (
            self.max_tokens
            if not hasattr(self.model, "max_seqlen")
            else self.model.max_seqlen
        )
        max_prompt_len = self.max_prompt_len or min(
            max_seqlen - self.max_gen_len, self.max_tokens - self.max_gen_len
        )
        prompts = [p[-max_prompt_len:] for p in prompts]
        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []
        it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        for batch in it:
            n_seqs = len(batch)
            generated_tokens = [[] for _ in range(n_seqs)]
            is_done = [False for _ in range(n_seqs)]
            packed_batch, lengths = pack_prompts(batch)
            packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
            n_seqs = lengths.size(0)

            # Prefilling cache
            prompt_logits = self.prefill(packed_batch.unsqueeze(0), lengths)
            # Selecting last token in each prompt
            all_tokens = sample_tokens(
                prompt_logits, self.temperature, self.top_p, self.top_k
            )
            start_token = all_tokens[:, lengths.cumsum(0) - 1]

            for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                generated_tokens[seq_id].append(tok)

            current_token = start_token
            for i in range(1, self.max_gen_len):

                next_logits = self.generate_next_token(current_token)
                next_token = sample_tokens(
                    next_logits.clone(), self.temperature, self.top_p, self.top_k
                )

                for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                    if not is_done[seq_id]:
                        generated_tokens[seq_id].append(tok)
                        current_end_str = self.tokenizer.decode(
                            generated_tokens[seq_id][-self.max_until_size :]
                        )
                        contains_end_string = any(
                            [e in current_end_str for e in self.until]
                        )
                        is_done[seq_id] = (
                            contains_end_string or tok == self.tokenizer.eos_id
                        )
                if all(is_done):
                    break

                current_token = next_token

            generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

            for p, logit in zip(
                batch, prompt_logits.squeeze(0).split(lengths.tolist()),
                strict=True,
            ):
                x = logit[:-1]
                y = torch.tensor(p[1:], device=x.device)
                loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                greedy.append((x.argmax(dim=-1) == y).cpu())

        return generation, loglikelihood, greedy

# TODO: Make this capable of distributed serving, per TorchTitan
def main():
    job_config = JobConfig()
    job_config.parse_args()

    mamba = load_model(
        job_config=job_config,
        checkpoint_path="/scratch/gpfs/mn4560/hazan-lab/tensorized_filters/tensorized_filters/models/mamba/log/model_01150.safetensors",
        models_config=models_config,
        model_name_to_cls=model_name_to_cls,
    )

    generator_config = PackedCausalMambaGeneratorConfig(
        temperature=0.7,
        top_p=0.95,
    )
    tokenizer = tiktoken.get_encoding("o200k_base")
    generator = PackedCausalMambaGenerator(generator_config, mamba, tokenizer)

    # Allow multiple prompts
    prompts = [
        """The capital of France is """,
    ]
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    # Start generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    # Calculate tokens per second
    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
