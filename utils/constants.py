"""System constants for the P/D-disaggregated LLM inference simulator."""

# GPU hardware profiles
# BYTES_PER_TOKEN is kept consistent (84,876 bytes) so profiling formulas remain valid.
# total_kv_cache_tokens = int(available_kv_memory_bytes / BYTES_PER_TOKEN)
GPU_PROFILES = {
    "A100_40GB": {
        "total_vram_gb": 40,
        "model_size_gb": 27,
        "total_kv_cache_tokens": 164458,   # int((40-27)*1024^3 / 84876)
    },
    "A100_80GB": {
        "total_vram_gb": 80,
        "model_size_gb": 27,
        "total_kv_cache_tokens": 670484,   # int((80-27)*1024^3 / 84876)
    },
}

DEFAULT_GPU_TYPE = "A100_40GB"

# Memory configuration (from simulation_rule.md) — defaults to A100 40GB
TOTAL_VRAM_GB = 40
MODEL_SIZE_GB = 27
TOTAL_KV_CACHE_TOKENS = 164458

# Memory in bytes
TOTAL_VRAM_BYTES = TOTAL_VRAM_GB * 1024**3
MODEL_SIZE_BYTES = MODEL_SIZE_GB * 1024**3
AVAILABLE_KV_MEMORY_BYTES = (TOTAL_VRAM_GB - MODEL_SIZE_GB) * 1024**3

# Memory fraction static (from simulation_rule.md)
MEM_FRACTION_STATIC = 0.9

# Scheduling configuration
SCHEDULE_CONSERVATIVENESS = 0  # --schedule-conservativeness 0
ENABLE_MIXED_CHUNK = True  # --enable-mixed-chunk

# Default scheduling limits
DEFAULT_MAX_PREFILL_TOKENS = 8192
DEFAULT_MAX_RUNNING_REQUESTS = 4096

# Token size (bytes per token for KV cache)
BYTES_PER_TOKEN = int(AVAILABLE_KV_MEMORY_BYTES / TOTAL_KV_CACHE_TOKENS)
