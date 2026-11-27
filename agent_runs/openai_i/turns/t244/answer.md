The last result shows that explicit routing required large attention changes; accurate dose transport alone did not preserve factual predictions. The neutral latent edit passed the local factual check, but its treatment response was too weak. I’ll recover the architecture and record-derived targets before testing a compensated route. It must pass both factual and rollout checks to be useful.

```python
from pathlib import Path
import numpy as np
import torch
import inspect

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())

notes218 = Path("/work/notes.md").read_text()
sections218 = notes218.split("\n## ")
print("\nEARLIER RECORD-EVIDENCE NOTES")
for s in sections218:
    if any(w in s.lower() for w in
           ("causal", "identif", "instrument", "policy", "proximal", "true effect")):
        if not any(f"Python {k}" in s[:90] for k in range(189, 218)):
            print(("## " + s)[:2200])
            if sum(len(x) for x in sections218[:sections218.index(s)+1]) > 90000:
                break

print("\nRELEVANT KERNEL OBJECTS")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    if any(w in name.lower() for w in
           ("203", "204", "209", "215", "216", "217", "innovation",
            "target", "causal", "replay", "rate", "prefix", "tuning_ids")):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif callable(obj):
            try:
                print(name, inspect.signature(obj))
            except (TypeError, ValueError):
                pass
        elif isinstance(obj, (float, int, str)):
            print(name, repr(obj)[:180])
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:12])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))
print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 218 — recover compensated-routing prerequisites\n"
            "- Neutral209 remains installed; its treatment-response columns are zero.\n"
            "- Reviewing architecture, record-derived effects, and retained helper contracts before another routing candidate.\n"
            "- No new weights changed. Any compensated route requires GPU factual and rollout validation.\n")
```