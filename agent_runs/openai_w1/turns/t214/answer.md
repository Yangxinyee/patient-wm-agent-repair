The recovered analyses support a negative insulin effect and a positive norepinephrine effect after adjusting for pre-dose history, with broadly consistent signs across measurement channels. These remain observational estimates, not known truth. The weight inspection stopped on a tensor-type mismatch. I’ll fix that and inspect the cached pathway construction; a weak latent projection would explain why the previous posterior edits did not affect predictions.

```python
import numpy as np
from pathlib import Path

print("LATENT/PRELUDE/HEAD WEIGHTS")
for name in NAMES:
    if name.startswith(("core.z_in", "core.z_head", "core.prelude.0.attn",
                        "core.prelude.0.self_attn", "attr_heads.value_bin")):
        w = np.asarray(load_original(name))
        print(name, w.shape, "rms", round(float(np.sqrt(np.mean(w*w))), 7),
              "max", round(float(np.max(np.abs(w))), 7))

print("\nRECENT CACHES")
for prefix in ("r188", "r187", "rp_", "r168", "r164"):
    found = []
    for k, v in list(globals().items()):
        if k.startswith(prefix):
            if hasattr(v, "shape"):
                found.append((k, tuple(v.shape)))
            elif isinstance(v, dict):
                found.append((k, list(v)[:15]))
            else:
                found.append((k, type(v).__name__))
    print(prefix, found[:25])

print("\nCARRIER/HEAD VARIABLES")
print([k for k in globals() if any(s in k.lower() for s in
      ("carrier", "physical", "kernel", "head5", "head_5", "contrast", "causal"))][:45])

print("\nNOTEBOOK SECTIONS")
text = Path("/work/notes.md").read_text()
for line in text.splitlines():
    if line.startswith("## ") and any(str(i) in line.split(" /")[0] for i in range(140, 177)):
        print(line)
print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 190–191\nRecovered pre-dose-history-adjusted associations: insulin negative, norepinephrine positive, with alternate channels broadly agreeing. Run 190 weight inspection failed because load_original returns torch tensors; run 191 converts explicitly to numpy. No weights changed.\n")
```
