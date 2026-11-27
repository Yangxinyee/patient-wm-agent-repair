The useful cached states and regressions are still present, but the long variable listing again hid the helper definitions. Nothing has been edited. I’ll narrow the inspection to those helpers and prior encoder checks. A systematic encoder mismatch would explain the failed reconstruction; if those checks are sound, I’ll stop that line of debugging and use direct forward validation.

```python
import builtins, types, inspect

print("=== NOTEBOOK HELPER SIGNATURES ===")
for nm, fn in builtins.list(globals().items()):
    if isinstance(fn, types.FunctionType) and fn.__module__ in ("__main__", None):
        if nm in {"load", "load_original", "save_edit", "drop_edit", "edited"}:
            continue
        print(nm, inspect.signature(fn), "uses", fn.__code__.co_names)
print("=== ENCODER CHECKS ===")
print(repr(encoder_checks)[:1700])
print("=== OBSERVATION DIAGNOSTICS ===")
print(repr(obs_diagnostics)[:2000])
print("=== STATE FILE FIELDS ===")
for nm, val in S.items():
    print(nm, val.shape, val.dtype)
print("=== FIRST STATE/RECORD ALIGNMENT ===")
print([(int(S["stay_id"][i]), float(S["hour"][i]), int(S["n_atoms"][i]),
        repr(bundle_atom_info[i])[:260]) for i in range(5)])
print("=== CURRENT BASELINE AND REFERENCE ===")
print("factual NLL", float(fact0.nll.mean()))
print("rollout", roll0.groupby("edit").shift.mean().to_dict() if False
      else roll0.groupby("edit")["shift"].mean().to_dict())
print("record-derived targets", matched_refs)
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 31; forwards 3. Cached original states and causal reference targets remain available. Narrow inspection of helper definitions and observation diagnostics. No edits.\n")
```