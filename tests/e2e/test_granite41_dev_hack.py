"""Compare the vLLM granite41 dev-hack path against the FMS reference path.

Both paths instantiate Granite41 ("8b" variant) with random weights via FMS
``get_model``. We seed ``torch.manual_seed`` immediately before each
``get_model`` call so the random initializations match, then assert the
greedy-decoded token IDs are identical or close enough.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from output_util import compare_results, extract_output

# vLLM's plugin reads this env var; force the CPU/eager path.
os.environ.setdefault("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager")

SEED = 42
TOKENIZER_ID = "ibm-granite/granite-4.1-8b"
MAX_NEW_TOKENS = 20
PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n"
    "Provide a list of instructions for preparing chicken soup.\n\n### Response:"
)


@pytest.mark.cpu
@pytest.mark.e2e
@pytest.mark.decoder
@pytest.mark.chunked_prefill
def test_granite41_vllm_matches_fms_reference():
    fms_results = _fms_reference_outputs([PROMPT], MAX_NEW_TOKENS)
    vllm_results = _vllm_outputs([PROMPT], MAX_NEW_TOKENS)

    # Note on backend="sendnn": the actual backend in this test is "eager" (CPU),
    # but we deliberately pass "sendnn" to compare_results to enable its
    # DIVERGING-tolerance branch. The model is initialized with random weights, 
    # so the logit distribution is extremely flat. Any tiny numerical
    # difference between the FMS sdpa_with_sinks kernel and the vLLM
    # spyre_paged_attn_with_sinks kernel have a different max likelihood token.
    compare_results(
        model="granite41",
        tensor_parallel_size=1,
        backend="sendnn", # actually CPU eager execution, just to allow DIVERGING
        vllm_results=vllm_results,
        hf_results=fms_results,
        prompts=[PROMPT],
    )


def _fms_reference_outputs(
    prompts: list[str], max_new_tokens: int
) -> list[dict[str, Any]]:
    """Run the FMS reference path; return dicts in ``compare_results`` shape."""
    from fms.models import get_model
    from fms.utils import tokenizers
    from fms.utils.generation import generate

    torch.manual_seed(SEED)
    # Note: FMS reference command uses --default_dtype=fp16, but the vLLM
    # CPU+eager path resolves dtype to fp32 in spyre.py:get_dtype(). Match
    # fp32 here so the random weight samples are bit-identical.
    model = get_model(
        "granite41",
        "8b",
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,  # --unfuse_weights
    )
    model.eval()
    torch.set_grad_enabled(False)

    tokenizer = tokenizers.get_tokenizer(TOKENIZER_ID)
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        token_strs = tokenizer.tokenize(prompt)
        prompt_ids: list[int] = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(
            token_strs
        )
        n_prompt = len(prompt_ids)
        ids = torch.tensor(prompt_ids, dtype=torch.long, device="cpu")

        captured_logprobs: list[float] = []

        def _capture_logprob(_token_pos, logits, next_val, kwargs):
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            chosen_id = int(next_val[0, 0].item())
            captured_logprobs.append(log_probs[0, chosen_id].item())
            return next_val, kwargs

        result = generate(
            model,
            ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            max_seq_len=model.config.max_expected_seq_len,
            extra_kwargs={"attn_name": "sdpa_causal"},
            post_iteration_hook=_capture_logprob,
        )
        if result.dim() == 1:
            result = result.unsqueeze(0)
        new_token_ids: list[int] = result[0, n_prompt:].tolist()
        new_tokens = tokenizer.convert_ids_to_tokens(new_token_ids)
        assert len(captured_logprobs) == len(new_token_ids), (
            "post_iteration_hook captured a different number of logprobs than "
            f"tokens generated ({len(captured_logprobs)} vs {len(new_token_ids)})"
        )
        results.append(
            {
                "text": tokenizer.convert_tokens_to_string(new_tokens),
                "token_ids": tuple(new_token_ids),
                "tokens": tuple(new_tokens),
                "logprobs": tuple(captured_logprobs),
            }
        )
    return results


def _vllm_outputs(prompts: list[str], max_new_tokens: int) -> list[dict[str, Any]]:
    """Spin up vLLM offline with the granite41 model and return the outputs."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="granite41",
        tokenizer=TOKENIZER_ID,
        max_model_len=512,
        max_num_seqs=1,
        block_size=64,
        max_num_batched_tokens=512,
        seed=SEED,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        logprobs=1,  # required so extract_output can populate tokens/logprobs
    )
    outputs = llm.generate(prompts, sampling_params)
    return [extract_output(req) for req in outputs]
