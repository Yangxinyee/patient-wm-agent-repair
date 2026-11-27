The recurrent error is about 9% of the exposed state’s scale—potentially useful for exploring directions, but not for certifying an edit. The earlier correlation-based directions failed. I’ll check the action encoding and recurrent helper before measuring finite-difference responses; inconsistent response signs would rule out this approach.

```python
import dis
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(8)
torch.set_grad_enabled(False)

print("=== RECORD AND EMBEDDING CONVENTIONS ===")
print(Path("/data/README.md").read_text()[:1800])
print(records.head(6).to_string(index=False))
print("unit convention:", unit146)
print("atom137:", atom137.__code__.co_varnames[:atom137.__code__.co_argcount],
      "defaults:", atom137.__defaults__)
print("event kinds:", VOCAB["event_kinds"], "atom kinds:", VOCAB["atom_kinds"])
for name in NAMES:
    if name.startswith("embed.categorical."):
        a = W[name]
        print(name, "row norms", torch.linalg.vector_norm(a, dim=1).numpy().round(3))

print("\n=== ACTION ATOM HELPER BYTECODE ===")
byte151 = list(dis.Bytecode(atom137))
for instruction in byte151[:220]:
    print(instruction.opname, instruction.argrepr)

print("\n=== RECURRENT STEP BYTECODE ===")
for instruction in list(dis.Bytecode(core_step136)):
    if instruction.opname not in {
        "RESUME", "CACHE", "PRECALL", "PUSH_NULL", "STORE_FAST",
        "LOAD_FAST", "LOAD_GLOBAL"
    }:
        print(instruction.opname, instruction.argrepr)

print("\n=== EARLIER HEAD CONVENTIONS ===")
for sec in sections149:
    if any(s in sec.splitlines()[0] for s in
           ["Python 16;", "Python 17;"]):
        print(sec[:650])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 151 — finite-difference preparation\n"
        "- Recurrent discrepancies are roughly 9% relative to state scale; "
        "not an exact reconstruction and not factual certification.\n"
        "- Checking action encoding and the update implementation before "
        "testing response-based rather than correlation-based directions.\n"
        "- No edits active.\n"
    )
```
