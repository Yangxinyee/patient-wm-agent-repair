The posterior check is accurate, but the full transition still has reconstruction error. I’ll treat anchored dose contrasts as approximate edit-design data, not causal evidence. I’ll also check the register conventions against the factual feature cache; inconsistent conventions would invalidate a decoder fit before any GPU test.

```python
import numpy as np, torch, pandas as pd
def to_np0(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("CURRENT CHECKPOINT")
for name0 in graft_names:
    print(name0, float(np.sqrt(np.mean(
        (to_np0(load(name0)) - to_np0(graft_checkpoint_beta2[name0]))**2))))
print("\nSAMPLED INPUT CACHES")
for name0, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        if obj.ndim == 3 and obj.shape[-1] == 192 and obj.shape[0] in (4, 8):
            print(name0, tuple(obj.shape))
    elif isinstance(obj, dict):
        matched0 = {k: tuple(v.shape) for k, v in obj.items()
                    if isinstance(v, (np.ndarray, torch.Tensor))
                    and v.ndim == 3 and v.shape[-1] == 192 and v.shape[0] in (4, 8)}
        if matched0:
            print(name0, matched0)
print("\nMETADATA AND STATES")
for name0, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and len(obj) in (4510, 4758, 4751):
        print(name0, obj.shape, list(obj.columns))
print("state_files", {k: shape_summary(v) for k, v in state_files.items()})
print("\nPARTS")
parts_probe0 = cpu_head_parts(lab_meta, states0)
print({k: shape_summary(v) for k, v in parts_probe0.items()})
print("\nREGISTER FEATURE CHECK")
repair_probe0 = repair_feature_sets(lab_meta, states0)["head_input"]
prior_probe0 = parts_probe0["prior"][:, :32]
plain_probe0 = ln(parts_probe0["base"] + lin(prior_probe0, "z_head")
                  + W["type_emb.weight"][1], "query_norm") + parts_probe0["additive"]
print("plain vs repaired RMS", float(np.sqrt(np.mean(
    (to_np0(plain_probe0) - to_np0(repair_probe0))**2))))
print("\nREADME\n" + open("/data/README.md").read()[:1800])
with open("/work/notes.md", "a") as handle:
    handle.write("\n## Python 154 recovery audit; forward 11/50\n"
                 "- Synthetic posterior is accurate; full transition reconstruction remains approximate.\n"
                 "- Use anchored contrasts only as edit-design data, never as evidence for true drug effects.\n"
                 "- Current checkpoint retained. Auditing register conventions and factual input caches.\n")
```
