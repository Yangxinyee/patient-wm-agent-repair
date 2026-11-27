The recovered lag estimates support effects lasting several hours, especially for insulin. That makes the new immediate-response directions useful probes, but not yet calibrated repair directions.

I’ll inspect the retained state arrays and posterior dimensions next. This will let me build a dose-innovation edit that is small on factual records, then test whether its rollout effect persists with the right sign.

```python
import inspect
from pathlib import Path

print("=== POSTERIOR AND LATENT TENSORS ===")
for _n in NAMES:
    if any(s in _n for s in ("z_post", "z_prior", "z_in", "z_head")):
        print(_n, SHAPES[_n])

print("\n=== RETAINED DATA / CALIBRATION OBJECTS ===")
_wanted = [
    "S", "W", "state_x", "state_df", "state_rows", "states", "train", "panel",
    "ctx152", "e152", "u152", "ecache152", "act_rows152", "obs_rows152",
    "latent_directions161", "jacobian161", "args161", "batch_args161",
    "eligible153", "chosen_stays153", "audit_indices161"
]
for _n in _wanted:
    if _n not in globals():
        continue
    _o = globals()[_n]
    print(_n, type(_o).__name__, end=" ")
    if isinstance(_o, dict):
        print("keys:", list(_o)[:16])
    elif hasattr(_o, "files"):
        print("files:", _o.files)
    elif hasattr(_o, "shape") and not callable(_o.shape):
        print("shape:", tuple(_o.shape))
        if hasattr(_o, "columns"):
            print("columns:", list(_o.columns))
    elif isinstance(_o, (list, tuple)):
        print("length:", len(_o), "first item type:", type(_o[0]).__name__ if _o else None)
    else:
        print(str(_o)[:180])

print("\n=== REMAINING GLOBAL NAMES ===")
print(" ".join(sorted(
    n for n in globals().copy()
    if not n.startswith("_") and n[0].lower() >= "r"
))[:1900])

print("\n=== REPLAY / READOUT HELPER DETAILS ===")
for _n in ("replay_inputs161", "call_core161", "readout153", "readout_capture156", "capture159"):
    if _n not in globals():
        continue
    _f = globals()[_n]
    print("\n", _n, inspect.signature(_f))
    try:
        print(inspect.getsource(_f)[:1500])
    except Exception:
        print("Names:", _f.__code__.co_names)
        print("Arguments/local names:", _f.__code__.co_varnames[:30])

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Resume: Python 162–164\n"
        "- Recovered the architecture and record-derived controlled lag estimates.\n"
        "- Best replay error is 2.5% relative h RMS; immediate latent response directions remain approximate probes.\n"
        "- No active edits. Next candidate must use factual-centred dose features and matched GPU validation.\n"
    )
```
