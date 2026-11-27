The recovered causal checks support the same immediate-effect signs, but the output was truncated before the reconstruction details. I’ll inspect those details without printing weight contents. In particular, I’ll check whether the gate coefficients were fitted on hidden features but applied to raw head inputs—an incompatible feature basis would explain their failure.

```python
import re
import dis

print("=== NOTEBOOK SECTION TITLES ===")
for m in list(re.finditer(r"(?m)^## .+$", notes_text))[:32]:
    print(m.group(0), "offset", m.start())

print("\n=== COMPACT HELPER BYTECODE ===")
for name in ("rh_eval", "rh_register", "wmlp", "wnorm"):
    f = globals()[name]
    print("\nFUNCTION", name, "locals", f.__code__.co_varnames)
    lines = [f"{i.opname} {i.argrepr}" for i in dis.get_instructions(f)]
    print("\n".join(lines)[:2200])

print("\n=== CONFIGURATION OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith(("wr_", "rec_", "mc_", "repair_")):
        if isinstance(obj, dict):
            summary = {}
            for k, v in list(obj.items())[:15]:
                summary[k] = (
                    {"shape": tuple(v.shape)} if hasattr(v, "shape")
                    else {"keys": list(v)[:10]} if isinstance(v, dict)
                    else repr(v)[:90]
                )
            print(name, summary)
        elif isinstance(obj, (str, int, float, tuple, list)):
            print(name, repr(obj)[:200])

print("\n=== GATE BASIS CHECK ===")
print("RMS(rec_head_X - wr_phi):",
      float(np.sqrt(np.mean((np.asarray(rec_head_X) - np.asarray(wr_phi))**2))))
print("gate prediction ranges:")
print(pd.DataFrame({
    "event": wr_meta.event.to_numpy(),
    "stored_gate": np.asarray(rec_gate_pred).reshape(-1),
    "raw_linear": np.asarray(rec_head_X) @ np.asarray(rec_gate_v).reshape(-1)
}).groupby("event").agg(["min", "max", "mean"]))

print("\n=== SMALL RECONSTRUCTION INPUTS ===")
print(wr_meta.head(2).to_string(index=False))
print("wr_nf first rows:", np.asarray(wr_nf)[:2])
print("wr_dt range:", float(wr_dt.min()), float(wr_dt.max()))
print("\n=== RELEVANT GLOBAL NAMES ===")
print([n for n in globals() if n.startswith(("rec_", "mc_", "wr_", "rh_"))])
```
