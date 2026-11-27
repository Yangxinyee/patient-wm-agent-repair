The earlier analysis found dose-innovation directions in model states, with record-based effects persisting across several hours. That suggests testing where those directions influence recurrent updates, rather than amplifying all inputs. I’ll finish reviewing prior edits before selecting a new probe; this avoids repeating changes that already failed factual validation.

```python
from pathlib import Path
import inspect

notes = Path("/work/notes.md").read_text()
print("EARLIER EDIT RESULTS")
print(notes[11000:18000])

print("\nTRANSFORM HELPERS")
for fn in ["apply_lowrank_left_transform", "apply_lowrank_right_transform",
           "effective_transform"]:
    if fn in globals():
        try:
            print(inspect.getsource(globals()[fn])[:1500])
        except Exception:
            print(fn, inspect.signature(globals()[fn]))

print("\nRECURRENT AND VALUE-HEAD TENSORS")
for name in NAMES:
    if name.startswith(("core.c_out", "core.b_in", "core.delta_in",
                        "core.a_in", "core.coda", "attr_heads.value_bin")):
        print(name, SHAPES[name])
```
