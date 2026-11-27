The decoder regression reduced disagreement with the original model, but worsened held-out factual NLL. I’m rejecting it without changing weights. Next I’ll examine a different mechanism: dose changes also alter the latent posterior, potentially changing inferred condition as well as action. A record-fitted replacement for that pathway is worth testing only if factual states remain accurate and dose responses move toward the identified effects.

```python
import re, numpy as np, torch
from pathlib import Path

print("POSTERIOR AND ACTION-ENCODER TENSORS")
for n183 in NAMES:
    if n183.startswith(("z_post", "encode_action", "z_in")):
        print(n183, SHAPES[n183])

print("\nAVAILABLE POSTERIOR / ACTION INPUTS")
shown183 = 0
for k183 in sorted(list(globals())):
    if re.search(r"(^core_|^cf_|^action_|posterior|zpost|^ctx|^Eall|^Uall|prelude|features)", k183):
        o183 = globals()[k183]
        if hasattr(o183, "shape"):
            print(k183, tuple(o183.shape))
        elif isinstance(o183, dict):
            print(k183, "dict", list(o183.keys())[:8])
        elif callable(o183) and hasattr(o183, "__code__"):
            c183 = o183.__code__
            print(k183, "function", c183.co_varnames[:c183.co_argcount],
                  c183.co_names[:12])
        elif isinstance(o183, (list, tuple)):
            print(k183, "list", len(o183))
        shown183 += 1
        if shown183 >= 65:
            print("Stopped after 65 relevant names.")
            break

print("\nPRELUDE TRAINING CONTEXT HISTORY")
notes183 = Path("/work/notes.md").read_text()
secs183 = re.split(r"(?=^## )", notes183, flags=re.M)
for s183 in secs183:
    if re.search(r"\b(?:44|45|46|47|48|49|50|51)\b", s183.split("\n", 1)[0]):
        print(s183[:430])

print("\nCURRENT EDITS", edited())
with Path("/work/notes.md").open("a") as f183:
    f183.write("\n## Run 183: reject hidden-head distillation; inspect posterior inference pathway\n"
               "Hidden-feature distillation reduced KL but all tested penalties increased "
               "stay-held-out NLL; no decoder edit installed. Inspecting whether a "
               "record-predictable action component can replace dose-dependent posterior "
               "inference while leaving the direct action pathway intact. No weights changed. "
               "Python runs: 183; forwards: 22.\n")
```
