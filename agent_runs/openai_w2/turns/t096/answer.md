The architecture confirms that observations and actions use separate encoders. The remaining mismatch could therefore be in either the observation attributes or its encoder. I’ll inspect those helpers and the vocabulary; finding a convention error should let us fix reconstruction without changing any weights.

```python
import dis, json, inspect
from pprint import pformat

print("=== VOCABULARY CONVENTIONS ===")
for key in ["config", "codes", "event_kinds", "atom_kinds", "numeric_note"]:
    print(key, pformat(VOCAB.get(key), width=105))
print("numeric_specs:", pformat(globals().get("numeric_specs"), width=105))
for name in ["code_ids", "unit_ids_repair", "rate_unit_ids_repair"]:
    print(name, pformat(globals().get(name), width=105))

print("\n=== RECONSTRUCTION HELPERS ===")
for name in ["embed_record_atom", "rmlp", "rnorm", "batched_set_repair"]:
    fn = globals().get(name)
    if fn is None:
        continue
    print("\nFUNCTION", name)
    try:
        print(inspect.getsource(fn))
    except (OSError, TypeError):
        print(dis.Bytecode(fn).dis())

print("\n=== OTHER LOCAL HELPER NAMES ===")
print([name for name, obj in list(globals().items())
       if inspect.isfunction(obj)
       and getattr(obj, "__module__", "") == "__main__"
       and any(s in name.lower() for s in
               ["post", "recon", "bundle", "encode", "state", "linear", "repair"])])
print("Current edits:", edited())
```