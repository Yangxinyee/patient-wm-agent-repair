The architecture shows that drug information enters through the action encoder, while reading predictions also rely on last-reading registers. The output was truncated before the state comparison, so there is no new evidence yet for an edit. I’ll retrieve that comparison and the earlier causal estimates; a dose-linked, low-variance state direction would support a selective amplification test.

```python
from pathlib import Path
import numpy as np

print("=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:3400])

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if (name.startswith("encode_action.")
        or name.startswith(("core.flow.", "core.a_in", "core.delta_in", "core.b_in",
                            "core.c_out", "core.z_", "z_in.", "z_head."))
        or "value_bin" in name or name.startswith("embed.bins.")
        or name.startswith("query_norm.")):
        print(name, SHAPES[name])

print("\n=== PAIRED STATE CHANGES ===")
for label, sl in [("out", slice(0,192)), ("h", slice(192,960)), ("z", slice(960,992))]:
    xx = state24_original["x"][:, sl].astype(float)
    dx = delta24[:, sl].astype(float)
    print(label, "delta RMS", round(float(np.sqrt(np.mean(dx**2))), 6),
          "factual centered RMS", round(float(np.sqrt(np.mean((xx-xx.mean(0))**2))), 6))
    for flag in ["has_insulin", "has_norepinephrine"]:
        mask = state24_original[flag].astype(bool)
        print(" ", flag, "rows", int(mask.sum()),
              "delta RMS", round(float(np.sqrt(np.mean(dx[mask]**2))), 6))

print("\n=== CAUSAL RESULTS AND BASELINE ROLLOUTS ===")
for name in ["baseline_rollouts", "dml_results", "dynamic_moments", "probe_targets"]:
    value = globals().get(name)
    print(name, repr(value)[:1800])

print("\n=== SAVED CANDIDATES ===")
print(list(repair_candidates))
print("\n=== DATA VARIABLES ===")
for name, value in sorted(list(globals().items())):
    if hasattr(value, "columns") and hasattr(value, "shape"):
        print(name, value.shape, list(value.columns)[:16])
print("\n=== CURRENT OPERATIONS ===")
print(ops())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 56\n"
            "- Retrieving paired-state diagnostics and compact causal/candidate evidence after run 55 output truncation.\n"
            "- No edits changed; forward passes used: 24.\n")
```