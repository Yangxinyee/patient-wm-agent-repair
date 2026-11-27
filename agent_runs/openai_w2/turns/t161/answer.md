The diagnostics show why gain scaling alone is costly: the stronger edit causes factual shifts that do not generally correct the original errors. A cached decoder using prelude FFN features has roughly twice the norepinephrine response-to-error efficiency of the current decoder. I’ll inspect how to inject that signal analytically; if its local steering can be reconstructed reliably, it offers a testable lower-cost route.

```python
import inspect

print("=== RECONSTRUCTION HELPERS ===")
helpers141 = []
for name141 in sorted(list(globals())):
    obj141 = globals()[name141]
    if callable(obj141) and getattr(obj141, "__module__", "") == "__main__":
        if any(s in name141.lower() for s in
               ["step", "prelude", "core", "coda", "head", "recon", "local", "fit", "feature"]):
            try:
                sig141 = str(inspect.signature(obj141))
            except Exception:
                sig141 = "?"
            print(name141, sig141)
            helpers141.append(name141)

print("=== AVAILABLE HELPER SOURCE ===")
source_budget141 = 4200
for name141 in ["prelude_features111", "recurrent_step", "local_core_step",
                "core_step119", "repair_head_means"]:
    obj141 = globals().get(name141)
    if callable(obj141):
        try:
            src141 = inspect.getsource(obj141)
            shown141 = src141[:source_budget141]
            print(name141, shown141)
            source_budget141 -= len(shown141)
        except Exception as exc141:
            print(name141, "source unavailable:", type(exc141).__name__)
        if source_budget141 <= 0:
            break

print("=== RELEVANT TENSORS ===")
for name141 in NAMES:
    if any(s in name141 for s in
           ["prelude.0.linear", "prelude.0.norm", "prelude_norm",
            "core.b_in", "core.delta_in", "core.c_out"]):
        print(name141, SHAPES[name141])

ffnfit141 = prelude_fits112["ffn"][0.3]
print("FFN FIT:", {k: (np.shape(v) if hasattr(v, "shape") else repr(v)[:700])
                   for k,v in ffnfit141.items()})
print("FACTUAL FFN FEATURES:", np.shape(prelude_hidden111))
print("SAMPLE INDICES:", sample_map128[:12])
print("LOCAL CACHED NAMES:",
      [n for n in sorted(globals()) if any(s in n.lower() for s in
       ["119", "120", "124", "126", "127", "j128", "h128", "meta128"])][:100])
```
