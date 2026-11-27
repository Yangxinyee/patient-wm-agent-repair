The record-based estimates consistently indicate insulin lowers the next glucose reading, while norepinephrine raises the next MAP reading. The insulin estimate is stable across dose and time subsets, but its rollout target also depends on persistence. I’ll verify the cached head inputs and decoder scaling before building a small, gated correction. Agreement between cached probabilities and reconstructed logits will confirm the implementation.

```python
import torch, numpy as np, pandas as pd, inspect, json
from pathlib import Path
torch.set_num_threads(8)

print("=== ORIGINAL BASELINES ===")
for k in (1, 2, 6):
    p = Path(f"/work/fwd_{k}/summary.json")
    print(f"forward {k}:", p.read_text()[:1800])
print("Record-derived rollout targets:", record_targets)

print("\n=== FIT ENTRY STRUCTURES ===")
for drug in ("insulin", "norepinephrine"):
    for entry in fits99[drug][:2]:
        print(drug, [summary102(x) for x in entry])
print("Raw feature projection error:",
      np.max(np.abs(features99["raw"]["projection"] - np.eye(192))))

wc103 = cache98["with_code"].detach().cpu().numpy()
mu103 = features99["raw"]["mean"]
sd103 = features99["raw"]["sd"]
xf103 = features99["raw"]["X"]
print("Standardized raw feature max error:",
      float(np.max(np.abs(xf103 - (wc103-mu103)/sd103))))

print("\n=== HEAD AND REGISTER HELPERS ===")
for name in ("rh_eval", "rh_register"):
    try:
        print(inspect.getsource(globals()[name])[:2800])
    except Exception as e:
        print(name, type(e).__name__)

W1103 = load_original("attr_heads.value_bin.0.weight").cpu()
b1103 = load_original("attr_heads.value_bin.0.bias").cpu()
W2103 = load_original("attr_heads.value_bin.2.weight").cpu()
b2103 = load_original("attr_heads.value_bin.2.bias").cpu()
with torch.no_grad():
    H103 = torch.nn.functional.gelu(
        cache98["with_code"] @ W1103.T + b1103
    ).cpu().numpy()

print("\nHidden feature quantiles:",
      np.quantile(H103, [0, .01, .5, .99, 1]))
centered_W2103 = W2103.numpy() - W2103.numpy().mean(axis=0, keepdims=True)
scores103 = H103.reshape(-1,192).std(axis=0) * np.linalg.norm(centered_W2103,axis=0)
small103 = np.argsort(scores103)[:12]
print("Lowest variable logit-contribution neurons:",
      [(int(j), round(float(scores103[j]),6),
        round(float(H103[:,:,j].mean()),5)) for j in small103])

print("\nVOCAB STRUCTURE:", list(VOCAB.keys()))
for key in VOCAB:
    if any(s in key.lower() for s in ("code", "norm", "value_bin")):
        obj = VOCAB[key]
        print(key, repr(obj)[:900])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 103: candidate preparation\n"
            "Recovered stable DML immediate slopes: insulin -1.99285 (SE 0.04554), "
            "NE +3.29011 (SE 0.09111). Insulin state innovation decoder is useful; "
            "NE decoder is not. Checking standardized-feature conversion and exact "
            "cached value-head evaluation, then selecting low-contribution neurons "
            "for an insulin-only gated correction. No edits installed.\n")
```
