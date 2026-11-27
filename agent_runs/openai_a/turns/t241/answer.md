The error is explained: the screening function returns one loss per reading, not a scalar. I’ll average those losses and compare regularization strengths and edit scales. These CPU screens can rank candidates, but only paired GPU factual and rollout tests can establish whether a stronger edit is acceptable.

```python
import pandas as pd

def screen_value_candidate(cache, w, b, scale):
    score = cache["x"] @ np.asarray(w, dtype=np.float32) + np.float32(b)
    gate_t = torch.from_numpy(np.asarray(cache["gate"], dtype=np.float32))
    score_t = torch.from_numpy(np.asarray(score, dtype=np.float32))
    tilt = (
        torch.nn.functional.gelu(gate_t + score_t)
        - torch.nn.functional.gelu(gate_t)
    ).numpy() * np.float32(scale)
    logits = cache["logits"] + tilt[..., None] * sc_rank
    probabilities = torch.softmax(torch.from_numpy(logits), -1).mean(0).numpy()
    losses = -np.log(np.maximum(
        probabilities[np.arange(len(cache["bins"])), cache["bins"]], 1e-30
    ))
    means = np.sum(probabilities * cache["centres"], axis=-1)
    return float(np.mean(losses)), means

_screen_fact = {k: selective_cache(v, 30.0) for k, v in fact_c.items()}
_screen_syn = {k: selective_cache(v, 30.0) for k, v in syn_c.items()}
stronger_screen = []

for key, fit in r3_candidates.items():
    if key[0] not in (0.1, 0.3, 1.0):
        continue
    scales = (4.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0)
    for scale in scales:
        row = {"lambda": key[0], "penalty": key[1], "scale": scale}
        for split, cache in _screen_fact.items():
            nll, _ = screen_value_candidate(cache, fit["w"], fit["b"], scale)
            row[split] = nll - float(cache["original_nll"])
        means = {}
        for multiplier, cache in _screen_syn.items():
            _, means[multiplier] = screen_value_candidate(
                cache, fit["w"], fit["b"], scale
            )
        row["double"] = float(np.mean(means[2.0] - means[1.0]))
        row["half"] = float(np.mean(means[0.5] - means[1.0]))
        row["max_cost"] = max(row[s] for s in _screen_fact)
        stronger_screen.append(row)

stronger_screen_df = pd.DataFrame(stronger_screen)
print("=== CURRENT FAMILY ===")
print(stronger_screen_df.query("penalty == 1 and `lambda` == 0.3")
      .round(5).to_string(index=False))
print("=== BEST LOCAL RESPONSES WITH MAX CPU COST < 0.045 ===")
print(stronger_screen_df.query("max_cost < 0.045")
      .sort_values("double").head(18).round(5).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart recovery and repaired screening; Python 211, forwards 21/50\n"
        "- Confirmed current four tensors exactly match stronger6_checkpoint.\n"
        "- Python 206 failed because cache_losses returns a per-reading vector; "
        "screening now explicitly averages losses.\n"
        "- Recovered insulin fuzzy-RD beta=-1.928709 and persistence=0.831840; "
        "fixed-record targets train -11.136, first-100 tuning -10.708, half +5.354.\n"
        "- NE flexible estimates around +3.3 remain observational.\n"
        "- CPU candidate screens are design evidence only; no weights changed.\n"
    )
```
