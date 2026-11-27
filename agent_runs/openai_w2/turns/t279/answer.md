The persistence estimates are fitted assumptions, not known simulator dynamics. The mapping check stopped because `load` returns a Torch tensor; no weights changed. I’ll correct that conversion and verify the factorization. A mismatch would rule out constructing the trial from these cached factors.

```python
import torch
import pandas as pd

def numpy231(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)

original231 = numpy231(load_original("core.b_in.weight"))
current231 = numpy231(load("core.b_in.weight"))
checks231 = {
    "current_vs_validated": np.max(np.abs(current231 - validated_B225)),
    "insulin_factor_error": np.max(np.abs(
        delta_ins218 - np.outer(V218[:, 0], W218[:, 0]))),
    "NE_factor_error": np.max(np.abs(
        delta_ne218 - np.outer(V218[:, 1], W218[:, 1]))),
    "mapped_base_error": np.max(np.abs(
        decoder_to_weight226(coef_base226) - W218[:, 0])),
    "validated_gain_error": np.max(np.abs(
        current231 - (original231 + 1.20*delta_ins218 + 1.05*delta_ne218))),
}
print("CHECKS", checks231)
assert max(checks231.values()) < 2e-5

print("\nBLEND DIAGNOSTICS")
print(blend_table226.to_string(index=False))
print("decoder argument names", decoder_to_weight226.__code__.co_varnames)
print("decoder global names", decoder_to_weight226.__code__.co_names)

ids231 = sorted(int(s) for s in all_tuning_ids183)
print("\nTUNING IDS", len(ids231), ids231[:4], ids231[-4:])
print("ROLLOUT COLUMNS", roll225.columns.tolist())
print("TRAIN SCHEMA", pd.read_csv("/data/train.csv", nrows=2).columns.tolist())

with open("/work/notes.md", "a") as f231:
    f231.write(
        "\n## Runs228–231: restart verification\n"
        "Architecture and causal-evidence notes recovered. Drug slopes and "
        "latent persistence remain record-based estimates with identification "
        "assumptions. No weight changes. Current validated_B225 reconstruction "
        f"checks: {checks231}. Preparing a controlled half-dose decoder trial.\n"
    )
```
