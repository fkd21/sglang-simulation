Project goal:
Build a simulation system for P/D-disaggregated LLM inference (X prefill Y decode).

Simulation requirements:
- The simulator must be based on real profiling formulas.
- The simulator must follow the real scheduling logic used in SGLang as closely as possible.
- The simulator must use real workload traces, including Azure Conv and Azure Code traces.
- The simulator should use these components as the baseline system, and then add new mechanisms on top of that baseline.

New mechanisms to implement:
1. Partial prefill offloading:
   - A prefill request can transfer part of its prompt to a decode instance for execution.
   - The transferred amount is beta * total_prefill_tokens.
   - beta is a tunable policy parameter.

2. Decode continuation on prefill instance:
   - A decode request can continue generation on a prefill instance.
   - The continuation length is M tokens.
   - M is a tunable policy parameter.

3. Role switching:
   - A prefill instance can switch to a decode instance.
   - A decode instance can switch to a prefill instance.
   - Switching should be modeled as a scheduling/control decision in the simulator.

Main objective:
- Use the simulator to find the best policy for autoscaling, work stealing, and role switching.
- Use the simulator to identify effective decision boundaries for when each new mechanism should be enabled.

Expected outputs:
- A configurable simulation framework.
- Support for policy search over beta, M, and switching conditions.
- Metrics that allow comparison between baseline P/D disaggregation and the new mechanisms.

Simulation mode:
- Use discrete-event simulation instead of fixed-step simulation.
- The simulator should advance time by processing events in chronological order.

1. Scheduling and queueing:
#TODO
- Read the real implementation in SGLang and extract the core scheduling and queueing mechanisms.
- Summarize how requests are admitted, batched, scheduled, migrated, and completed.
- server setting should --enable-mixed-chunk and --schedule-conservativeness 0
- Identify the main data structures, scheduling rules, and control logic that determine prefill/decode behavior.
- The simulator should reproduce these core mechanisms as faithfully as possible, while allowing extensions for new policies.



2.Use the following regression model to estimate one-iteration mixed-batch inference time:

t_inference = 0.009076
            + 7.063e-5 * total_prefill_tokens
            + 5.623e-5 * decode_batch_size
            + 6.926e-8 * decode_computed_token_sum

Where:
- total_prefill_tokens: total number of prefill tokens in the current batch
- decode_batch_size: number of decode requests in the current batch
- decode_computed_token_sum: total computed decode-context tokens in the current batch
- The time unit is s.


3.Use the following linear model to estimate KV transfer latency:

t_kv_transfer = 1.86683e-6 * total_prefill_tokens

Where:

- total_prefill_tokens: total number of prefill tokens whose KV cache is transferred
- The time unit is s.

4. memory and model size:

    Memory allocation must follow the real allocation mechanism as closely as possible.

    40 GB VRAM,  model size 8B.
    TOTAL_KV_CACHE_TOKENS = 164458 # Llama 3.1 8B on A100 40G
    Model wights=27GB
    --mem-fraction-static 0.9


    







