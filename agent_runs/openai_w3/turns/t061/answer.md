The pilot barely changed the dose responses and moved doubled insulin slightly in the wrong direction. That refutes the statistical latent direction as an effective causal write direction, despite its preserved factual fit. I’ll inspect the actual latent-to-state and state-to-value computations next. A computed direction must produce a substantial rollout change before I consider it useful.

```python
import numpy as np
import torch
import pandas as pd

print("=== CORE TENSORS ===")
for n in NAMES:
    if n.startswith("core."):
        print(n, SHAPES[n])
print("\n=== RELEVANT OTHER TENSORS ===")
for n in NAMES:
    if any(n.startswith(p) for p in (
        "query_norm", "type_emb", "last_numeric", "last_status",
        "z_prior", "z_in", "z_head", "gap", "embed_transition.value")):
        print(n, SHAPES[n])
print("\n=== CONFIG ===", VOCAB["config"])

print("\n=== CHANNEL MAGNITUDES ===")
for n in ("z_in.weight", "z_head.weight", "z_post.2.weight",
          "core.prelude.0.self_attn.in_proj_weight",
          "attr_heads.value_bin.0.weight"):
    z = as_numpy(load_original(n))
    print(n, "Frobenius:", float(np.linalg.norm(z)),
          "column norms:", np.round(np.quantile(
              np.linalg.norm(z, axis=0), [0,.1,.5,.9,1]), 5))
print("state h scale:",
      np.quantile(as_numpy(sx[:,192:960]).std(axis=0), [0,.1,.5,.9,1]))
print("state z scale:",
      np.quantile(as_numpy(sx[:,960:]).std(axis=0), [0,.1,.5,.9,1]))

repair_factual_original = pd.read_csv("/work/fwd_4/factual_original.csv")
repair_factual_edited = pd.read_csv("/work/fwd_4/factual_edited.csv")
print("\n=== PRECISE FACTUAL DIFFERENCES ===")
print("columns:", list(repair_factual_original.columns))
for col in repair_factual_original.select_dtypes(include=np.number).columns:
    if col not in ("stay_id", "hour"):
        delta = (repair_factual_edited[col].to_numpy() -
                 repair_factual_original[col].to_numpy())
        print(col, "mean delta:", float(delta.mean()),
              "RMS:", float(np.sqrt(np.mean(delta**2))),
              "max absolute:", float(np.max(np.abs(delta)))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 54; forwards 5. Pilot factual value NLL unchanged at "
        "reported precision, but dose readouts barely moved: insulin x2 "
        "-.406 vs original -.422; half .199 vs .213; NE x2 -.726 vs -.742; "
        "half -.230 vs -.238. Statistical reading-z directions are ineffective "
        "for causal writes. Inspecting exact core/head transfer before replacing "
        "the direction; current pilot remains provisional.\n"
    )
```