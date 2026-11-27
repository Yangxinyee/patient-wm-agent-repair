The mask test resolved the mismatch: “true” marks valid memory entries, and the reconstructed output now matches the forward tool closely. I’ll recover the record-based effect estimates and validated helper functions before computing an edit. A useful correction must respond to dose changes without materially changing factual predictions.

```python
from pathlib import Path
import inspect

arch76 = Path("/model/ARCHITECTURE.md").read_text()
notes76 = Path("/work/notes.md").read_text()
print("=== ARCHITECTURE ===")
print(arch76)
print("\n=== EARLY CAUSAL EVIDENCE ===")
print(notes76[:11000])
print("\n=== RETAINED FUNCTION INTERFACES ===")
funcs76 = []
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            funcs76.append((name, str(inspect.signature(obj))))
        except Exception:
            pass
for name, sig in funcs76:
    print(name, sig)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 76 / mask resolved\n"
            "True denotes valid memory entries. With that convention, one-step out RMS "
            "is 1.51e-5 and h RMS is .00251. Recovering causal targets and validated "
            "interfaces before computing a factual-preserving dose correction. No edits.\n")
```