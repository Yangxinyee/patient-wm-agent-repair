The early estimates suggest a negative insulin effect, but its magnitude varies by method; norepinephrine estimates are even less stable. Those results alone are not enough to set the edit. I’ll inspect the later identification checks and GPU trials. Agreement across better-controlled record analyses would support a target; sensitivity to the controls would argue for caution.

```python
import re
from pathlib import Path

text = Path("/work/notes.md").read_text()
sections = re.split(r"(?=^## )", text, flags=re.M)
wanted = [
    "Recovery and threshold identification",
    "Sustained insulin target",
    "Cross-stay dose-policy fitting",
    "Gated raw-input insulin graft",
    "Latent-robust gated graft",
    "GPU validation of latent-robust graft",
    "Stronger insulin probe rejected",
]
for label in wanted:
    hits = [s for s in sections if s.startswith("## " + label)]
    for section in hits:
        print(section[:1200])
        if len(section) > 1200:
            print("[section truncated; total characters:", len(section), "]")

print("=== RETAINED EDIT PARAMETERS ===")
for key in ["graft_beta", "graft_alpha", "graft_gate_scale", "graft_unit",
            "graft_nodes", "graft_names", "graft_cpu_penalty", "graft_applied",
            "beta", "causal_shift_table", "ins_target", "matched_causal_targets163"]:
    if key in globals():
        val = globals()[key]
        if hasattr(val, "shape") and getattr(val, "size", 1000) > 40:
            print(key, type(val).__name__, val.shape)
        else:
            print(key, repr(val)[:900])
```
