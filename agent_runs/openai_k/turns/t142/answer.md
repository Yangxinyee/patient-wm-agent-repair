The correction channels are separate, but the underlying posterior was changed substantially. Before calibrating their downstream effects, I’ll test whether each channel actually tracks changes in its own drug’s physical rate. A dose decoder that saturates, reverses sign, or mixes drugs would explain the failed rollout and would not generalize to stopping or tripling doses.

```python
import numpy as np
import torch
import dis
import re
from pathlib import Path

r131_notes = Path("/work/notes.md").read_text()
print("=== POSTERIOR CONSTRUCTION NOTES ===")
r131_start = r131_notes.find("## Run 64")
r131_end = r131_notes.find("## Run 68", r131_start)
print(r131_notes[r131_start:r131_end][:3400])

print("\n=== CORE / ENCODER DEPENDENCIES ===")
for r131_n in ["dataset80", "r59_core", "prior_features74",
               "rr35_scaled_action_encoding", "repaired_core_local"]:
    r131_f = globals().get(r131_n)
    if r131_f is not None:
        print(r131_n)
        print("  globals:", r131_f.__code__.co_names)
        print("  locals:", r131_f.__code__.co_varnames[:20])

print("\n=== CORE WRAPPER IMPLEMENTATION ===")
r131_f = globals().get("r59_core")
if r131_f is not None:
    for r131_i in dis.get_instructions(r131_f):
        if r131_i.opname not in ["RESUME", "CACHE", "EXTENDED_ARG"]:
            print(r131_i.opname, r131_i.argrepr)

print("\n=== RETAINED INPUT / CALIBRATION ARRAYS ===")
for r131_n in sorted(list(globals())):
    if re.search(r"(66|88)$", r131_n) or r131_n in [
        "hflow_audit", "e_audit", "u_audit", "out_audit",
        "W", "weights", "diagnostics90"
    ]:
        r131_o = globals()[r131_n]
        if isinstance(r131_o, (np.ndarray, torch.Tensor)):
            print(r131_n, tuple(r131_o.shape))
        elif isinstance(r131_o, dict):
            print(r131_n, "keys", list(r131_o)[:10])

print("\n=== BASELINE ACTION LATENT DIAGNOSTICS ===")
for r131_n in ["diagnostics90", "post66_rate_models", "rr39_policy_scores"]:
    r131_o = globals().get(r131_n)
    if isinstance(r131_o, dict):
        for r131_k, r131_v in r131_o.items():
            if isinstance(r131_v, dict):
                print(r131_n, r131_k, "keys", list(r131_v)[:12])
            elif isinstance(r131_v, (np.ndarray, torch.Tensor)):
                print(r131_n, r131_k, "shape", tuple(r131_v.shape))
            else:
                print(r131_n, r131_k, repr(r131_v)[:700])

with open("/work/notes.md", "a") as r131_file:
    r131_file.write(
        "\n## Run 131: audit physical-dose decoding before rollout calibration\n"
        "- Correction channels occupy latent coordinates 30 and 31; their token directions have norms 3.867 and 5.438.\n"
        "- Posterior first-layer changes are large, so dose monotonicity and drug separation require direct verification.\n"
        "- Python runs: 131; forward passes: 6; no submitted edits.\n"
    )
```