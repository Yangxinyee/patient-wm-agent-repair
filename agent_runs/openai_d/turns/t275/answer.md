The actual factual NLL increase is 0.00477, narrowly passing this check. I’ll save this as a tested fallback, then inspect the norepinephrine correction for a direction that costs less factual accuracy. A useful replacement must strengthen both norepinephrine readouts without weakening insulin or exceeding the NLL limit.

```python
import numpy as np
import pandas as pd
import torch

r235_safe = {n: load(n).clone() for n in edited()}
np.savez("/work/safe_6288c226d6.npz",
         **{n: t.cpu().numpy() for n, t in r235_safe.items()})

print("=== EDITED NEURONS ===")
for rc_name in sorted(r235_safe):
    rc_a = r235_safe[rc_name].cpu().numpy()
    rc_b = load_original(rc_name).cpu().numpy()
    rc_diff = np.abs(rc_a-rc_b)
    if rc_a.ndim == 2:
        rc_axis = 1 if ".0." in rc_name else 0
        rc_ix = np.where(np.max(rc_diff, axis=rc_axis) > 1e-7)[0]
        print(rc_name, rc_a.shape, "changed hidden indices:", rc_ix.tolist())
    else:
        print(rc_name, rc_a.shape,
              "changed entries:", np.where(rc_diff > 1e-7)[0].tolist())

print("\n=== CACHE AND CANDIDATE NAMES ===")
rc_names = [
    n for n in globals()
    if n.startswith(("r201", "c207", "c210", "r230", "r231"))
    or n == "cache231"
]
for rc_name in rc_names:
    rc_obj = globals()[rc_name]
    if hasattr(rc_obj, "shape"):
        print(rc_name, tuple(rc_obj.shape))
    elif isinstance(rc_obj, dict):
        print(rc_name, "dict:", list(rc_obj.keys())[:12])
    elif inspect.isfunction(rc_obj):
        print(rc_name, inspect.signature(rc_obj))
    elif isinstance(rc_obj, (int, float, str)):
        print(rc_name, str(rc_obj)[:120])

print("\n=== EVALUATOR DETAILS ===")
for rc_name in ["c207_tensor_weights", "c210_eval", "r201_eval",
                "r201_fixed", "c211_logits"]:
    rc_obj = globals().get(rc_name)
    if inspect.isfunction(rc_obj):
        print(rc_name, inspect.signature(rc_obj),
              "locals:", rc_obj.__code__.co_varnames,
              "names:", rc_obj.__code__.co_names)

print("\n=== NE DEVELOPMENT SUMMARY ===")
rc_s = rc_notes.find("## Run 175")
rc_e = rc_notes.find("## Run 212")
rc_part = rc_notes[rc_s:rc_e] if rc_s >= 0 else ""
rc_keep = [
    s for s in rc_part.splitlines()
    if s and not s.startswith((" ", "{", "}", "[", "]"))
]
print("\n".join(rc_keep)[-2800:])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Forward 29 / 30 and Run 235: validated fallback\n"
            "Current edit hash 6288c226d6: tuning 0–99 rollout shifts "
            "insulin -13.693/+6.196; NE +1.262/-0.633. "
            "Tuning 100–199 factual NLL 1.62521 vs original 1.62044, "
            "delta +0.00477. Saved exact tensors in r235_safe and "
            "/work/safe_6288c226d6.npz. Insulin is near the provisional "
            "record-derived target; NE remains underpowered. "
            "No weights changed during this audit.\n")
```