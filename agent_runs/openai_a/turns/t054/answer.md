The value head has 17 outputs; the vocabulary lists only the 16 reading-bin centres, so the missing-value centre must be added explicitly. I’ll inspect the retained feature code and the norepinephrine evidence. A residual-dose correction is worth testing only if the state encodes dose variation beyond what the recorded history predicts.

```python
import dis
import re
from pathlib import Path

print("=== FEATURE HELPER BYTECODE ===")
fn = repair_head_features
for instruction in dis.get_instructions(fn):
    if instruction.opname not in ("RESUME", "CACHE", "PRECALL"):
        print(instruction.opname, instruction.argrepr)

print("\n=== REGISTER EXAMPLES ===")
print(reg_check[:4])
print("last_numeric column norms:",
      torch.linalg.vector_norm(load_original("last_numeric.weight"), dim=0).tolist())

print("\n=== NOTEBOOK SECTION SUMMARIES ===")
notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)^## ", notes)
for section in sections:
    if any(word in section.lower()
           for word in ("norepinephrine", "ne ", "threshold", "discontinu", "placebo")):
        print("\n##", section[:850])

print("\n=== STATE FILE STRUCTURE ===")
for key, obj in state_files.items():
    print(key, type(obj).__name__)
    if hasattr(obj, "keys"):
        print([(k, getattr(obj[k], "shape", None)) for k in obj.keys()])
```
