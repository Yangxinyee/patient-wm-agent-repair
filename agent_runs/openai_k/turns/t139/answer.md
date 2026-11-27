The records support an insulin effect near −2 at one hour and a norepinephrine effect near +3, with different persistence. The failed candidate changed only the latent-to-state projection; its zero “drug templates” do not establish zero drug effects. I’ll inspect how that candidate was constructed and recover the original rollout responses before choosing a better-separated correction.

```python
import dis, io
from pathlib import Path
import pandas as pd
import numpy as np
import torch

print("=== RELEVANT TENSOR SHAPES ===")
for r128_n in NAMES:
    if (r128_n.startswith(("z_", "encode_action.", "attr_heads.value_bin.",
                          "last_numeric.", "core.flow."))
        or r128_n in ["embed.numeric.weight", "embed.numeric.bias"]):
        print(r128_n, SHAPES[r128_n])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for r128_k in ["codes", "config", "normalization", "normalisation"]:
    if r128_k in VOCAB:
        print(r128_k, repr(VOCAB[r128_k])[:1800])

print("\n=== CANDIDATE CONSTRUCTION ===")
for r128_n in ["build_candidate89", "hook_mlp90"]:
    r128_f = globals().get(r128_n)
    if r128_f is not None:
        r128_buf = io.StringIO()
        dis.dis(r128_f, file=r128_buf, depth=0)
        print(r128_n, "\n", r128_buf.getvalue()[:3200])

print("\n=== CANDIDATE-RELATED OBJECTS ===")
r128_names = [n for n in globals()
              if any(s in n.lower() for s in
                     ["candidate", "template", "direction", "innovation", "policy"])]
for r128_n in sorted(r128_names)[:65]:
    r128_o = globals()[r128_n]
    if isinstance(r128_o, dict):
        print(r128_n, "dict keys:", list(r128_o)[:12])
    elif hasattr(r128_o, "shape"):
        print(r128_n, type(r128_o).__name__, tuple(r128_o.shape))
    elif isinstance(r128_o, (list, tuple)):
        print(r128_n, type(r128_o).__name__, "length", len(r128_o))

print("\n=== OFFICIAL ORIGINAL ROLLOUT SUMMARY ===")
r128_paths = sorted(Path("/work").glob("fwd_*/dose_edit_original.csv"))
if r128_paths:
    r128_path = r128_paths[-1]
    r128_roll = pd.read_csv(r128_path)
    print("file:", r128_path, "shape:", r128_roll.shape)
    print(r128_roll.groupby("edit")["shift"].agg(["mean", "count"]).to_string())
    print("hour range:", r128_roll["hour"].min(), r128_roll["hour"].max())

with open("/work/notes.md", "a") as r128_file:
    r128_file.write(
        "\n## Run 128: candidate construction and baseline rollout audit\n"
        "- Record impulse estimates: insulin/glucose -2.073 at one hour, NE/MAP +2.952; glucose persists longer.\n"
        "- Inspecting the latent projection candidate and official baseline rollout before another edit.\n"
        "- Python runs: 128; forward passes: 6; no submitted edits.\n"
    )
```