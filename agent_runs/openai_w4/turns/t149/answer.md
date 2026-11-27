The edit still passes the tested factual-fit check, but its insulin rollouts are asymmetric and its norepinephrine response is small. Those predictions do not establish the true sustained effects. I’ll recover the architecture and retained diagnostics before deciding whether the asymmetry comes from the repair or the recorded dose distribution. A useful repair must match record-supported dynamics, not just the immediate regression coefficients.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE (read first) ===")
print(Path("/model/ARCHITECTURE.md").read_text())

def brief132(obj, depth=0):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return f"{type(obj).__name__}{tuple(obj.shape)}"
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        return f"{type(obj).__name__}{obj.shape}, columns={list(obj.columns)[:18]}"
    if isinstance(obj, dict):
        if depth >= 2:
            return f"dict keys={list(obj)[:24]}"
        return {str(k): brief132(v, depth + 1) for k, v in list(obj.items())[:24]}
    if isinstance(obj, (list, tuple)):
        if len(obj) <= 12 and all(np.isscalar(v) for v in obj):
            return obj
        return f"{type(obj).__name__}, len={len(obj)}, first={brief132(obj[0], depth+1) if obj else None}"
    if callable(obj):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(signature unavailable)"
        return f"{type(obj).__name__}{sig}"
    return repr(obj)[:400]

print("\n=== RETAINED REPAIR AND DYNAMICS OBJECTS ===")
for name in [
    "repair_bank", "chosen_channel116", "directions118",
    "weights_channel125", "weights_rejected125",
    "scale_results128", "nll_scales129",
    "kernel_models88", "provisional_kernels"
]:
    if name in globals():
        print(name, brief132(globals()[name]))

print("\n=== REPLAY INTERFACES ===")
for name in [
    "replay_factual125", "transition73",
    "transition_with_b110", "probe_means95"
]:
    fn = globals().get(name)
    if callable(fn):
        print(name, brief132(fn))
        print("defaults:", brief132(fn.__defaults__))
        print("referenced names:", fn.__code__.co_names)
        try:
            source = inspect.getsource(fn)
            print(source[:7000])
        except (OSError, TypeError):
            print("Notebook source unavailable through inspect.")

print("\n=== RELEVANT RETAINED NAMES ===")
print(sorted(
    name for name in list(globals())
    if not name.startswith("_")
    and any(s in name.lower() for s in [
        "kernel", "causal", "innovation", "replay", "rollout",
        "nll", "source", "target", "policy", "dose", "tuning"
    ])
))

print("\n=== EARLIER EVIDENCE NOTES ===")
notes132 = Path("/work/notes.md").read_text()
early132 = notes132[:-8000] if len(notes132) > 8000 else notes132
paragraphs132 = early132.split("\n\n")
selected132 = [
    p for p in paragraphs132
    if any(s in p.lower() for s in [
        "causal", "kernel", "instrument", "discontinu",
        "confound", "innovation", "original rollout",
        "dose_edit", "ground truth", "identif", "cross-fit"
    ])
]
print("\n\n".join(selected132)[-14000:])

print("\n=== CURRENT EDIT MAGNITUDES ===")
for name in edited():
    delta = np.asarray(load(name)) - np.asarray(load_original(name))
    print(name, "shape", delta.shape,
          "delta_norm", float(np.linalg.norm(delta)),
          "max_abs", float(np.max(np.abs(delta))))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 132\n"
        "Recovered architecture, retained replay/repair interfaces and earlier "
        "identification evidence. Current GPU rollouts remain predictions, not "
        "truth; immediate DML coefficients alone do not establish sustained "
        "intervention effects. No weights changed.\n"
    )
```