The separate batch worsened by 0.00613 nats per reading, above the limit. The candidate is therefore not reliably validated. I’ll inspect the paired losses and scaling helper before backing off; the next candidate must preserve useful dose responses while leaving a clearer factual-loss margin.

```python
import inspect
import dis
import numpy as np
import pandas as pd

print("=== SCALING HELPER ===")
print("make242 signature:", inspect.signature(make242))
dis.dis(make242)

print("\n=== RELEVANT OBJECTS ===")
for audit246_name in [
    "candidate243", "frontier242", "head177_names", "head177_free",
    "mapcoef242", "mapdir242", "cal242", "trainmc242", "tunemc242",
    "base_train242", "base_tune242"
]:
    audit246_obj = globals().get(audit246_name)
    if isinstance(audit246_obj, pd.DataFrame):
        print(audit246_name, audit246_obj.shape)
        print(audit246_obj.head(5).to_string(index=False))
    elif isinstance(audit246_obj, dict):
        print(audit246_name, {
            k: (getattr(v, "shape", None) or type(v).__name__)
            for k, v in list(audit246_obj.items())[:12]
        })
    else:
        print(audit246_name, type(audit246_obj).__name__,
              getattr(audit246_obj, "shape", None),
              str(audit246_obj)[:180] if not hasattr(audit246_obj, "shape") else "")

print("\n=== PAIRED GPU CSV SCHEMA ===")
audit246_o = pd.read_csv("/work/fwd_28/factual_original.csv")
audit246_e = pd.read_csv("/work/fwd_28/factual_edited.csv")
print(audit246_o.head(2).to_string(index=False))
print("columns:", audit246_o.columns.tolist())
assert len(audit246_o) == len(audit246_e)
audit246_nll_cols = [
    c for c in audit246_o.columns
    if "nll" in c.lower() and pd.api.types.is_numeric_dtype(audit246_o[c])
]
for audit246_c in audit246_nll_cols:
    audit246_delta = audit246_e[audit246_c] - audit246_o[audit246_c]
    print(audit246_c, "mean delta:", audit246_delta.mean(),
          "sum delta:", audit246_delta.sum())

print("\n=== LOCAL STRENGTH FRONTIER ===")
print(pd.read_csv("/work/independent_strength_frontier242.csv")
      .sort_values("tuning_dNLL").head(12).to_string(index=False))

with open("/work/notes.md", "a") as audit246_f:
    audit246_f.write(
        "\n## Run 246: independent validation failure\n"
        "- Forward 25 / fwd_26: candidate243 factual dNLL +0.00456 on tuning 0–99.\n"
        "- Forward 26 / fwd_27: rollouts insulin -10.198/+5.585, NE +1.059/-1.872.\n"
        "- Forward 27 / fwd_28: independent tuning 200–299 factual dNLL +0.00613, above tolerance.\n"
        "- Candidate243 is not reliably validated; recovering scaling helper before reducing factual cost.\n"
        "- No weights changed in this run.\n"
    )
```