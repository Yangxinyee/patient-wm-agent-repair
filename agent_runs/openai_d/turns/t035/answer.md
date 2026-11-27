The cached reconstruction and causal-analysis functions are still available; the output overflow came from printing a function’s large default weight dictionary. No new evidence changes the working hypothesis. I’ll inspect compact cache summaries next, then use them to test whether reading-head features retain dose innovations.

```python
print("=== FUNCTIONS (PARAMETER NAMES ONLY) ===")
for nm in sorted(list(globals())):
    ob = globals()[nm]
    if not nm.startswith("_") and callable(ob) and getattr(ob, "__module__", None) == "__main__":
        try:
            print(nm, "(" + ", ".join(inspect.signature(ob).parameters) + ")")
        except Exception:
            pass

print("\n=== RELEVANT CACHES ===")
for nm in sorted(list(globals())):
    if nm.startswith("_"):
        continue
    ob = globals()[nm]
    if any(k in nm.lower() for k in
           ["state", "head", "feat", "prior", "align", "config", "factual", "target", "rh_", "rc_"]):
        if hasattr(ob, "shape"):
            print(nm, type(ob).__name__, tuple(ob.shape))
        elif isinstance(ob, dict):
            print(nm, "keys", list(ob)[:10])
        elif isinstance(ob, (list, tuple)):
            print(nm, "len", len(ob), str(ob)[:120])

print("\n=== SELECTED TENSORS ===")
for nm in NAMES:
    if (nm.startswith(("attr_heads.value", "last_numeric", "last_status",
                       "embed_transition.value", "query_norm", "type_emb",
                       "core.z_", "core.gap", "core.flow", "encode_action"))
        or nm in ["embed.numeric.weight", "embed.numeric.bias"]):
        print(nm, SHAPES[nm])

print("\n=== FORWARD FILES ===")
for fld in sorted(Path("/work").glob("fwd_*")):
    print(fld.name, [p.name for p in fld.iterdir()])

print("\n=== RECONSTRUCTION FUNCTION REFERENCES ===")
for nm in ["rh_register", "rh_eval", "rc_lin", "rc_mlp", "rc_norm", "response_arrays"]:
    fun = globals()[nm]
    print(nm, "variables:", fun.__code__.co_varnames,
          "globals:", fun.__code__.co_names,
          "constants:", str(fun.__code__.co_consts)[:450])
```
