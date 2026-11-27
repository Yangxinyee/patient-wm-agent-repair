The additional states are available, giving us a second independent training batch. Next I’ll verify the decoder’s raw-feature scaling and recover the metadata builder needed to pool the batches. If the corrected predictions match the earlier diagnostics, the scaling issue is resolved; otherwise I’ll refit from explicit physical-dose targets.

```python
import pandas as pd, inspect

state_files[5] = dict(np.load("/work/fwd_5/states_original.npz", allow_pickle=False))
print("=== METADATA BUILDER ===")
print(inspect.signature(reading_metadata))
print("locals:", reading_metadata.__code__.co_varnames)

print("\n=== RETAINED RECORD TABLES ===")
for name, value in sorted(list(globals().items())):
    if isinstance(value, pd.DataFrame) and "stay_id" in value.columns and "event" in value.columns:
        print(name, value.shape, list(value.columns)[:16])

raw_hidden_train = HX * hidden_scale[None, :] + hidden_location[None, :]
raw_hidden_tune = HTX * hidden_scale[None, :] + hidden_location[None, :]
print("\n=== CORRECTED DECODER PREDICTIONS ===")
for drug, dec in dose_decoders.items():
    for label, features in (("train", raw_hidden_train), ("tuning", raw_hidden_tune)):
        pred = features @ dec["w"] + dec["b"]
        print(drug, label, "mean/sd", float(pred.mean()), float(pred.std()),
              "quantiles", np.quantile(pred, [.01, .5, .99]).tolist())

print("\n=== FEATURE BUILDER OUTPUT CONTRACT ===")
try:
    print(inspect.getsource(make_head_features))
except (OSError, TypeError):
    import dis
    instructions = list(dis.get_instructions(make_head_features))
    for ins in instructions[-24:]:
        print(ins.opname, ins.argrepr)

print("\n=== CACHED TARGET-RELATED ARRAYS ===")
for name, value in sorted(list(globals().items())):
    if isinstance(value, (np.ndarray, torch.Tensor)) and any(
        s in name.lower() for s in ("target", "sens", "tilt", "variance", "resid", "y_")
    ):
        print(name, tuple(value.shape))
print("\nNew states:", {k: v.shape for k, v in state_files[5].items()})

with open("/work/notes.md", "a") as f:
    f.write("\n## Additional training states; Python 57, forward 5/50\n"
            "- Obtained all bundle states for training stays 1010000100..1010000199.\n"
            "- Correcting standardized/raw feature conventions before any weight edit.\n")
```