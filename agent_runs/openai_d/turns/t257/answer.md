The caches contain full head inputs for independent training and tuning cohorts. Before adding a centering term, I’ll check whether the norepinephrine correction is firing before any norepinephrine has run. Suppressing such activation could save factual likelihood without affecting the scored hours. This explanation is refuted if the excess loss is already concentrated after exposure.

```python
import pandas as pd
import torch.nn.functional as F

print("=== CACHE METADATA ===")
for label, cc in [("train", c210_train), ("tune100", c210_tune)]:
    print(label, list(cc["meta"].columns))
    print(cc["meta"].head(3).to_string(index=False))

print("\n=== EXISTING GUARD / DECODER OBJECTS ===")
for key in ["r157_gate", "gates99", "repair_decoder_fits",
            "repair_decoder_cv", "dml_cache"]:
    obj = globals().get(key)
    if isinstance(obj, dict):
        print(key, {
            str(k): (list(v.keys()) if isinstance(v, dict)
                     else tuple(v.shape) if hasattr(v, "shape")
                     else type(v).__name__)
            for k, v in obj.items()
        })

print("\n=== EVALUATOR RETURN STRUCTURE ===")
res221 = c210_eval(c206_candidates["combined_strong"], c210_tune)
print(type(res221).__name__)
if isinstance(res221, dict):
    print({k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:250])
           for k, v in res221.items()})
elif isinstance(res221, (tuple, list)):
    print([(type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else repr(v)[:250])
           for v in res221])

print("\n=== PAIRED-NEURON SIGNALS BY EVENT AND HOUR ===")
for label, cc in [("train", c210_train), ("tune100", c210_tune)]:
    X = torch.as_tensor(cc["X"], dtype=torch.float32)
    ww = c207_tensor_weights(c206_candidates["combined_strong"])
    meta = cc["meta"]
    outrows = []
    for drug, pair in [("insulin", [107, 109]), ("norepinephrine", [48, 171])]:
        a = F.linear(X, ww["attr_heads.value_bin.0.weight"][pair],
                     ww["attr_heads.value_bin.0.bias"][pair])
        signal = (F.gelu(a[..., 0]) - F.gelu(a[..., 1])).mean(0).numpy()
        common = a.mean(-1).mean(0).numpy()
        for event in sorted(meta["event"].unique()):
            for lo, hi in [(-100, 12), (12, 100)]:
                mask = ((meta["event"].to_numpy() == event)
                        & (meta["hour"].to_numpy() >= lo)
                        & (meta["hour"].to_numpy() < hi))
                if mask.any():
                    outrows.append({
                        "cache": label, "drug": drug, "event": event,
                        "hours": f"{lo}..{hi}", "n": int(mask.sum()),
                        "signal_mean": float(signal[mask].mean()),
                        "signal_sd": float(signal[mask].std()),
                        "common_mean": float(common[mask].mean()),
                        "active_fraction": float((common[mask] > 0).mean())
                    })
    print(pd.DataFrame(outrows).round(4).to_string(index=False))

print("\n=== AVAILABLE RECORD / PANEL VARIABLES ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if isinstance(obj, pd.DataFrame) and any(
        s in key.lower() for s in ["train", "panel", "hour", "record", "tune"]
    ):
        print(key, obj.shape, list(obj.columns)[:20])
```
