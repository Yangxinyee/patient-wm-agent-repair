The original model’s training factual NLL is 1.63841, close to its tuning result. That gives us a second preservation check. I’ll now assemble the inputs needed to reproduce the reading head, including its previous-reading register and latent prior. Agreement with the saved means and bin NLLs will confirm the implementation.

```python
import pandas as pd
import torch.nn.functional as F
torch.set_num_threads(8)

W0 = {n: torch.as_tensor(load_original(n), dtype=torch.float32).clone() for n in NAMES}

print("=== LATENT, GAP, AND ENCODER NAMES ===")
for n in NAMES:
    if any(s in n for s in ("z_prior", "z_post", "z_head", "z_in", "gap.")) or n.startswith("encode."):
        print(n, SHAPES[n])

fact_train0 = pd.read_csv("/work/fwd_4/factual_original.csv")
print("\n=== FACTUAL FILE ===")
print(fact_train0.columns.tolist())
print(fact_train0.head(8).to_string(index=False))
print("\n=== FIRST STAY EVENTS ===")
print(train[train.stay_id == 1010000000].head(13).to_string(index=False))

print("\n=== LAST-NUMERIC COLUMN NORMS ===")
print(torch.linalg.vector_norm(W0["last_numeric.weight"], dim=0).numpy())
print("\n=== CACHE DETAILS ===")
for drug in ("insulin", "norepinephrine"):
    cc = dml_cache[drug]
    print(drug, {k: getattr(v, "shape", type(v).__name__) for k, v in cc.items()})
    if "models" in cc:
        print(str(cc["models"])[:450])

print("\n=== OTHER POSSIBLY USEFUL TENSORS ===")
for n in NAMES:
    if not n.startswith(("embed.", "core.", "encode.", "encode_action.", "attr_heads.")):
        print(n, SHAPES[n])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 4\nOriginal factual predictions on the 100 training stays with saved states: NLL 1.63841 over 4510 readings. By event: finger 1.4350, lab 1.3445, arterial 1.5024, cuff 1.9645. No edits. Intended for validating a local reconstruction of the reading head.\n")
```
