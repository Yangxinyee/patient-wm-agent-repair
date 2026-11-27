The deterministic reconstruction now agrees with the GPU output to rounding precision. This validates the time flow, reading type, gap scaling, and register layout. I’ll verify that agreement on all cached rows and recover the dose-innovation targets next. Any correction must also work with sampled latent draws, not just the deterministic head.

```python
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

base97 = bases96["seconds"] + mlp96(
    gap_inputs96["log_seconds/10"], "gap", "gelu")
q97 = norm96(base97 + W96["type_emb.weight"][1], "query_norm")
wc97 = q97 + fixed96
logits97 = mlp96(wc97, "attr_heads.value_bin", "gelu") + trans96
prob97 = logits97.softmax(-1)
mean97 = (prob97 * centres96).sum(-1)
nll97 = -logits97.log_softmax(-1).gather(1, obin96[:, None]).squeeze(1)

print("=== FULL DETERMINISTIC VALIDATION ===")
for label97, error97 in [("mean", mean97 - tm96), ("NLL", nll97 - tn96)]:
    print(label97, "RMSE", float(error97.square().mean().sqrt()),
          "max absolute error", float(error97.abs().max()))
for event97 in meta96["event"].unique():
    mask97 = torch.tensor(meta96["event"].eq(event97).to_numpy())
    print(event97, "mean RMSE",
          float((mean97[mask97] - tm96[mask97]).square().mean().sqrt()),
          "NLL RMSE",
          float((nll97[mask97] - tn96[mask97]).square().mean().sqrt()))

print("\n=== PRIOR TENSORS ===")
for name97 in NAMES:
    if "prior" in name97:
        print(name97, SHAPES[name97])

print("\n=== TARGET AND ALIGNMENT CACHES ===")
for name97 in ["dml_cache", "dml_results", "impulse_results",
               "repair_aligned", "head_meta", "head_records",
               "repair_phi", "repair_decoder_fits", "repair_decoder_cv",
               "trial_active_decoder", "trial_full_classifier",
               "trial_active", "trial_y", "trial_groups",
               "repair_state_blocks", "record_targets"]:
    obj97 = globals().get(name97)
    if hasattr(obj97, "columns"):
        print(name97, tuple(obj97.shape), "columns:", list(obj97.columns))
    elif isinstance(obj97, dict):
        print(name97, "dict keys:", repr(list(obj97.keys())[:15]))
        for k97, v97 in list(obj97.items())[:3]:
            print(" ", k97, type(v97).__name__,
                  tuple(v97.shape) if hasattr(v97, "shape") else repr(v97)[:170])
    elif hasattr(obj97, "shape"):
        print(name97, tuple(obj97.shape))
    else:
        print(name97, type(obj97).__name__, repr(obj97)[:180])

print("\n=== CONTROLLED-RESPONSE NOTES ===")
notes97 = Path("/work/notes.md").read_text()
start97 = notes97.find("## Run 5")
stop97 = notes97.find("## Run 9", start97)
print(notes97[start97:stop97][:4500])

assert float((mean97 - tm96).abs().max()) < 0.001
assert float((nll97 - tn96).abs().max()) < 0.001
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 97: exact deterministic head recovered\n"
        "Validated all cached forward-8 readings: seconds-based time flow, "
        "gap(log(dt_seconds)/10), GELU, type embedding 1, and probe-resolved "
        "register. This removes the reconstruction uncertainty that invalidated "
        "earlier gated edits. Original weights remain unchanged.\n"
    )
```
