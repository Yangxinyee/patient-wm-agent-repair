The last output was truncated before the channel diagnostics, so it does not establish whether doubled doses are being compressed. The current edit passes the measured factual check but gives a small insulin response. I’ll recover the architecture and retained interfaces compactly: saturation would favor fixing the dose channel; a linear channel would point toward its downstream dynamics.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

arch144 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch144[:7000])
if len(arch144) > 7000:
    print("ARCHITECTURE REMAINDER:", len(arch144) - 7000, "characters")

print("\n=== RETAINED FUNCTION INTERFACES ===")
for name144 in [
    "transition73", "transition_with_b110", "replay_factual125",
    "probe_means95", "make_actions73", "build_actions73"
]:
    obj144 = globals().get(name144)
    if callable(obj144):
        print(name144, inspect.signature(obj144))
        print("dependencies:", obj144.__code__.co_names)

def compact144(obj, depth=0):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {
            str(k): compact144(v, depth + 1) if depth < 1 else
            ({"shape": tuple(v.shape)} if hasattr(v, "shape") else type(v).__name__)
            for k, v in list(obj.items())[:24]
        }
    if isinstance(obj, (list, tuple)):
        return {"length": len(obj), "first": compact144(obj[0], depth + 1) if obj else None}
    if isinstance(obj, (str, int, float, bool, np.number)):
        return str(obj)[:250]
    return type(obj).__name__

print("\n=== REPAIR OBJECTS ===")
for name144 in [
    "chosen_channel116", "directions118", "repair_bank",
    "lp_new136", "lp_new137", "scale_results128", "nll_scales129"
]:
    if name144 in globals():
        print(name144, compact144(globals()[name144]))

print("\n=== RELEVANT RETAINED NAMES ===")
names144 = sorted(
    k for k in globals()
    if any(s in k.lower() for s in [
        "kernel", "target137", "lp137", "channel", "action",
        "innov", "decode", "dose", "bootstrap"
    ])
    and not k.startswith("_")
)
print("\n".join(names144[:100]))
print("\nEDITED:", edited())

with open("/work/notes.md", "a") as f144:
    f144.write(
        "\n## Run 144\n"
        "Compact recovery of architecture, channel parameters and retained interfaces. "
        "Run-143 truncation did not diagnose dose compression. No weights changed.\n"
    )
```