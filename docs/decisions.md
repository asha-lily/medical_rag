# Decision log


### ADR 001: Use qwen3:8b instead of gpt-oss:20b as the RAGAS judge

**Date:** 2026-10-09  
**Status:** Accepted

**Context**
RAGAS needs an LLM judge to score faithfulness and answer relevancy. The judge runs locally through Ollama on a 16GB Apple M2, and it has to come from outside the Llama family because the generator is `llama3.2`. My first choice, `gpt-oss:20b`, crashed during the baseline run with a Metal out-of-memory error (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Activity Monitor showed memory almost full.

**Options considered**
1. Keep `gpt-oss:20b` and cut memory use (smaller context window, one worker, a quantised KV cache, closing other apps). This keeps the stronger judge, but its weights alone take up most of the 16GB.
2. Switch to `qwen3:8b`. It's less than half the size and from a different family, but it's a weaker judge.
3. Use a hosted judge such as Claude Haiku via `langchain-anthropic`. This removes the memory limit, but I rejected it mainly because of cost. A second reason: a local judge mirrors what I'd want in production, where I wouldn't send sensitive medical data to a third-party API.

**Evidence**

| Judge | Size (`ollama list`) | Settings | Baseline run (5 answered questions) |
|---|---|---|---|
| `gpt-oss:20b` | 13 GB | context 20000, 2 workers, reasoning "low" | Crashed at ~315 s/it, 0 scores |
| `qwen3:8b` | 5.2 GB | context 9000, 1 worker, reasoning off | 5/5 scored, no NaNs |

I didn't try option 1. The model was already close to the memory limit when it loaded, so a smaller model seemed more reliable than tuning settings.

**Decision**
I will use `qwen3:8b` as the RAGAS judge, with reasoning off, a 9000-token context window and one worker.

**Consequences**
- Positive: evaluation runs to completion on my laptop, and the judge is still independent of the generator.
- Negative: the judge is likely weaker on subtle faithfulness calls, and turning reasoning off may make it less careful. The baseline RAGAS scores come from a judge I haven't checked against human labels. A smaller context window raises the risk that inputs get silently cut off, which would lower faithfulness. With one worker, runs are slower.
- Revisit if: `qwen3:8b` agrees poorly with my hand labels, NaN scores start to appear, or a hosted judge or more memory becomes an option.
