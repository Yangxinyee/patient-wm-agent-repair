The retained candidate is a small, rank-one change to the value head’s final layer; it leaves the recurrent model untouched. I’ll install it reversibly and check actual factual predictions first. A GPU NLL increase above 0.005 would reject it, regardless of its estimated treatment response.

```python
import torch.nn.functional as F
import pandas as pd

def find_vector186(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        a = np.asarray(obj.detach().cpu() if isinstance(obj, torch.Tensor) else obj)
        return [a] if a.shape == (193,) else []
    if isinstance(obj, dict):
        return sum((find_vector186(v) for v in obj.values()), [])
    if isinstance(obj, (tuple, list)):
        return sum((find_vector186(v) for v in obj), [])
    return []

vectors186 = find_vector186(fits183[1.0])
assert len(vectors186) == 1, (type(fits183[1.0]), len(vectors186))
vector186 = vectors186[0].astype(np.float64).copy()

reconstructed186 = (
    F.linear(fact_x183[0], W["attr_heads.value_bin.2.weight"],
             W["attr_heads.value_bin.2.bias"]) + fact_trans183
)
error186 = float((reconstructed186 - fact_logits183[0]).abs().max())
print("Final-layer reconstruction maximum error:", error186)
assert error186 < 1e-4, "Features are not the final-layer inputs; do not install."

delta_w186 = np.outer(tilt_np183, vector186[:192]).astype(np.float32)
delta_b186 = (tilt_np183 * vector186[192]).astype(np.float32)
np.savez("/work/candidate186.npz", vector=vector186,
         delta_weight=delta_w186, delta_bias=delta_b186)

for name, delta in [
    ("attr_heads.value_bin.2.weight", delta_w186),
    ("attr_heads.value_bin.2.bias", delta_b186),
]:
    original = np.asarray(load_original(name), dtype=np.float32)
    save_edit(name, original + delta)

print("Approximate validation:", evaluate_head183(vector186))
print("Active edits:", edited())
tuning_stays186 = sorted(pd.read_csv("/data/tuning.csv")["stay_id"].unique().tolist())[:100]
print("Tuning stay IDs:", tuning_stays186)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 186 — install alpha-1 direct-head candidate\n"
            "- Verified cached features exactly reconstruct final-layer logits.\n"
            "- Installed rank-one ordinal corrections to value_bin.2 weight and bias only; saved candidate186.npz.\n"
            "- Approximate held dNLL +.00387 is not certification. Next: matched GPU factual and rollout tests.\n")
```