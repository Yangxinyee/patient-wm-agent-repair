The rollback is complete. The inspection output was overwhelmed by a function’s weight-dictionary default, so it did not reveal the useful details. I’ll inspect bounded function definitions instead. This is diagnostic only: it should tell us whether finite-dose transport can be tested from saved states without relying on the unvalidated local likelihood reconstruction.

```python
def bounded_definition(fn, limit):
    print(fn.__name__, "arguments",
          fn.__code__.co_varnames[:fn.__code__.co_argcount],
          "names", fn.__code__.co_names)
    try:
        text = inspect.getsource(fn)
    except Exception:
        bb = io.StringIO()
        dis.dis(fn, file=bb)
        text = bb.getvalue()
    print(text[:limit])

bounded_definition(drug_atom205, 3500)
bounded_definition(set205, 2600)
for nm in ["head_audit", "history_audit", "encoder_audit"]:
    print(nm, repr(globals()[nm])[:700])
print("STATE SCHEMA", [(k, S[k].shape) for k in S.files])
print("HEAD ROW COLUMNS", list(head_rows.columns))
print(head_rows.head(2).to_string(index=False))
print("CORE SHAPES",
      [(k, SHAPES[k]) for k in NAMES
       if k.startswith(("core.prelude.", "core.prelude_norm", "z_post.", "z_in."))])
```
