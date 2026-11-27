The earlier dose-response estimates are reasonably consistent across sensors, but rollout targets still depend on causal and persistence assumptions. The recent work prepared an action-rate decoder without installing it. I’ll inspect that decoder and the local recurrence helpers next; failure to distinguish dose changes reliably would rule out using it as a causal channel.

```python
import dis, inspect
import numpy as np
import torch

print("=== HELPER SIGNATURES ===")
for n in [
    "action_atom_exact", "action_hidden105", "channel_targets105",
    "action_vectors4", "prelude_all83", "local_step67",
    "wm_step_saved", "wm_layer", "wm_attention", "wm_lin",
    "next_means_crn74", "compact_function109"
]:
    f = globals().get(n)
    if callable(f):
        print(n, inspect.signature(f))

print("\n=== ACTION HELPER IMPLEMENTATIONS ===")
for n in ["action_hidden105", "channel_targets105"]:
    f = globals()[n]
    print("\n", n)
    try:
        print(inspect.getsource(f))
    except (OSError, TypeError):
        txt = dis.Bytecode(f).dis()
        print(txt[:3100])

print("\n=== WEIGHT / STATE CONTAINERS ===")
for n, v in list(globals().items()):
    if n.startswith("_"):
        continue
    if isinstance(v, dict) and any(
        isinstance(k, str) and (k == "out" or k == "h" or "core.c_out" in k)
        for k in v
    ):
        print(n, "dict keys", list(v)[:8])
    elif isinstance(v, np.lib.npyio.NpzFile):
        print(n, "npz keys", v.files)
print("\n=== RESERVED COORDINATES AND DECODER ===")
print("reserved103", reserved103)
print("rate_scale105", rate_scale105, "caps105", caps105)
print("best105 report", best105[2])
print("channel_coef105 shape", channel_coef105.shape)
print("channel_intercept105", channel_intercept105)
print("\n=== ACTION ENCODER / PRELUDE SHAPES ===")
for n in NAMES:
    if n.startswith("encode_action.") or n.startswith("core.prelude.0.attn.") or n.startswith("core.prelude_norm."):
        print(n, SHAPES[n])
```
