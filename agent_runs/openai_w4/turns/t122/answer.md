The compensated candidate barely changes local reading predictions, which is encouraging. It also suggests that the original recurrent update passes very little new bundle information to the reading head. I’ll inspect the state scales and update gains next; this will show whether the new signal needs a stronger connection into memory.

```python
print("--- state scales ---")
for label109, arr109 in (
    ("out", baseline108[:, :192]),
    ("h", baseline108[:, 192:960]),
    ("z", baseline108[:, 960:]),
):
    a109 = np.asarray(arr109)
    print(label109, "RMS", float(np.sqrt(np.mean(a109**2))),
          "across-bundle centred RMS",
          float(np.sqrt(np.mean((a109-a109.mean(axis=0))**2))),
          "max", float(np.max(np.abs(a109))))

print("\n--- recurrent gain tensors ---")
for n109 in ("core.delta_in", "core.a_in", "core.flow.a",
             "core.b_in.weight", "core.c_out.weight",
             "core.prelude_norm.weight", "core.out_norm.weight"):
    a109 = np.asarray(rW[n109])
    print(n109, a109.shape, "min/median/max",
          np.quantile(a109, [0, .5, 1]).round(6),
          "RMS", float(np.sqrt(np.mean(a109**2))))

print("\n--- saved capture return structure ---")
for n109 in ("Eslots102", "Aparts103", "Eparts102", "A102", "V102",
             "indices102", "raw104", "candidate_hidden104",
             "original_post104", "candidate_post104"):
    o109 = globals().get(n109)
    if hasattr(o109, "shape"):
        print(n109, type(o109).__name__, tuple(o109.shape))
    elif isinstance(o109, dict):
        print(n109, "keys", list(o109))
    elif isinstance(o109, (list, tuple)):
        print(n109, type(o109).__name__, "length", len(o109),
              "first type", type(o109[0]).__name__ if len(o109) else None)
    else:
        print(n109, repr(o109)[:120])

print("\n--- transition implementation ---")
import dis
dis.dis(transition73)
```
