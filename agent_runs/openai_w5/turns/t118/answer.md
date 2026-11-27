Clearing four coordinates increased factual NLL by only 0.00034, leaving most of the allowed budget. That supports testing a private dose channel, but does not yet establish a repair. Next I’ll inspect the encoder helpers and rate bookkeeping so the channel can be fitted without gradients and checked across finite dose changes.

```python
import inspect
import dis
import numpy as np

print("=== VOCAB CONFIGURATION ===")
print("keys", list(VOCAB))
print("config", VOCAB["config"])
for key in VOCAB:
    if "norm" in key.lower():
        print(key, repr(VOCAB[key])[:1600])

print("=== ACTION-ENCODER HELPER DEPENDENCIES ===")
for name in ["action_atom_exact", "set_encode_single_local", "wm_set", "action_vectors4"]:
    fn104 = globals()[name]
    print(name, "arguments", fn104.__code__.co_varnames[:fn104.__code__.co_argcount])
    print("uses", fn104.__code__.co_names)
    try:
        text104 = inspect.getsource(fn104)
    except Exception:
        text104 = dis.Bytecode(fn104).dis()
    print(text104[:1400])

print("=== RATE BOOKKEEPING ===")
print("acttable4 columns", list(acttable4.columns))
print(acttable4.head(4).to_string(index=False)[:1300])
for name in ["records4", "rows4", "R4", "rate55", "dose_actual4", "active83", "startmask4"]:
    obj104 = globals().get(name)
    if obj104 is None:
        continue
    shape104 = getattr(obj104, "shape", None)
    print(name, type(obj104).__name__, shape104)
    if isinstance(obj104, np.ndarray):
        print("sample", obj104[:6])
        if obj104.ndim == 2 and obj104.shape[1] == 2:
            print("means/max", obj104.mean(axis=0), obj104.max(axis=0))

print("=== SINGLE-ACTION ENCODER CHECK ===")
for drug104, grid104 in [
    ("insulin", np.array([0., 0.5, 2., 8., 16., 32., 48.])),
    ("norepinephrine", np.array([0., 0.01, 0.1, 0.5, 1., 2., 3.])),
]:
    atoms104 = action_atom_exact(drug104, grid104)
    encoded104 = set_encode_single_local(atoms104, "encode_action")
    print(drug104, "atom shape", tuple(atoms104.shape),
          "encoded shape", tuple(encoded104.shape),
          "finite", bool(np.isfinite(array103(encoded104)).all()))
print("EDITS", edited())
```