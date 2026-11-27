The architecture confirms that dose changes can act through the prelude and recurrent state, bypassing the ineffective latent pathway. I’ll inspect that implementation and the retained effect estimates. This will tell us whether a dose-innovation correction can be inserted where it actually influences predictions.

```python
import dis
import inspect
import pandas as pd

print("=== PRELUDE IMPLEMENTATION ===")
print(dis.Bytecode(r66_prelude).dis())

print("=== ENCODER IMPLEMENTATIONS ===")
for name in ("encode_one_set", "repair_action_vectors"):
    fn76 = globals()[name]
    print(name, inspect.signature(fn76))
    print(dis.Bytecode(fn76).dis()[:2600])

print("=== CORE TENSORS ===")
for name in NAMES:
    if name.startswith("core."):
        print(name, SHAPES[name])

print("=== EFFECT ESTIMATES ===")
for k, v in kernel_refs.items():
    print(k, repr(v)[:1000])

print("=== AVAILABLE TABLES ===")
for k, v in list(globals().items()):
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, list(v.columns)[:24])

print("=== PRELUDE / RESPONSE ARRAYS ===")
for k, v in list(globals().items()):
    if k.startswith(("r45_", "r66_", "r70_")):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:8])
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:120])

print("probe columns:", r67_probe_columns)
print("edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 76; forwards 7. Architecture confirms effective action-token "
        "prelude/recurrent path. Inspecting its exact operations and retained "
        "record-derived response fits before designing a dose-innovation edit. "
        "No weights edited.\n"
    )
```