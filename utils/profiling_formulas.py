"""Profiling formulas for inference time and KV transfer latency.

Based on regression models from simulation_rule.md.
"""


def compute_inference_time(
    total_prefill_tokens: int,
    decode_batch_size: int,
    decode_computed_token_sum: int
) -> float:
    """Compute one-iteration mixed-batch inference time.

    Regression model:
    t_inference = 0.014654
                + 6.944e-5 * total_prefill_tokens
                + 6.024e-5 * decode_batch_size
                + 8.463e-8 * decode_computed_token_sum

    Args:
        total_prefill_tokens: Total number of prefill tokens in the batch
        decode_batch_size: Number of decode requests in the batch
        decode_computed_token_sum: Total computed decode-context tokens

    Returns:
        Inference time in seconds
    """
    return (
        0.014654
        + 6.944e-5 * total_prefill_tokens
        + 6.024e-5  * decode_batch_size
        + 8.463e-8  * decode_computed_token_sum
    )


def compute_kv_transfer_time(total_prefill_tokens: int) -> float:
    """Compute KV cache transfer latency.

    Linear model:
    t_kv_transfer = 1.86683e-6 * total_prefill_tokens

    Args:
        total_prefill_tokens: Total number of prefill tokens whose KV cache is transferred

    Returns:
        Transfer time in seconds
    """
    return 1.86683e-6 * total_prefill_tokens
