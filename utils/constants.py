"""System constants for the P/D-disaggregated LLM inference simulator."""

# Memory configuration (from simulation_rule.md)
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
