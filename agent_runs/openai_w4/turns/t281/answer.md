Stricter gates reduce factual loss, but thresholds 50 and 60 give identical results, suggesting they switch off the factual insulin channel entirely. I will test the less extreme gain-1.5, threshold-40 candidate on the GPU. Its CPU loss is within budget; success requires a stronger doubling response without losing the halving response.

```python
import json
import pandas as pd
import torch

r246_backup = {n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in edited()}
r246_key = (1.5, 40.0)
r246_candidate = r245_candidates[r246_key]

for name in ("z_post.0.weight", "z_post.0.bias"):
    save_edit(name, r246_candidate[name])

print("Saved insulin innovation gain / total gate shift:", r246_key)
print("CPU screen:", r245_results[r246_key])
print("Edited tensors:", edited())

r246_factfile = "/work/fwd_22/factual_original.csv"
r246_ids = sorted(pd.read_csv(r246_factfile)["stay_id"].astype(int).unique().tolist())
print("Validation stay count:", len(r246_ids))
print("FACTUAL REQUEST:")
print(json.dumps({"type": "factual", "model": "both", "split": "tuning", "stays": r246_ids}))
print("DOSE REQUEST:")
print(json.dumps({"type": "dose_edit", "model": "both", "split": "tuning", "stays": r246_ids}))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 246\n"
            "Saved insulin innovation-only gain 1.5 / total paired-gate shift 40 "
            "from Run-245 candidate. Only z_post.0.weight and z_post.0.bias changed "
            "relative to the prior submission; backup in r246_backup. "
            "CPU dNLL .0037233; GPU factual and rollout validation pending. "
            "Stricter 50/60 gates have identical factual scores and likely silence "
            "the factual channel; selected the less extreme threshold 40.\n")
```