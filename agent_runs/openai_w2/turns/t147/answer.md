The recovered notes support an insulin effect around −1.55 to −1.97 per rate unit and a norepinephrine effect around +3.5 to +3.9, with uncertain accumulation over time. The recurrent dose decoder is ready; I’ll check its units and injection location so the first edit has an interpretable strength.

```python
import inspect

print("=== PERSISTENCE AND ROLLOUT ESTIMATES ===")
for marker in ["## Run 38:", "## Run 39:", "## Run 40:"]:
    start = notes_recover.find(marker)
    if start >= 0:
        end = notes_recover.find("\n## ", start + 4)
        print(notes_recover[start:end if end >= 0 else start + 1800][:1800])

print("\n=== EXACT EDIT INGREDIENTS ===")
for name in ["Wdose128", "bdose128", "V128", "A128", "Edecoder128",
             "E_linear_dose117", "E_linear_dose119", "b117", "b119"]:
    obj = globals().get(name)
    if obj is not None:
        arr = np.asarray(obj)
        print(name, arr.shape, "norm", float(np.linalg.norm(arr)),
              "values" if arr.size <= 8 else "column norms",
              arr if arr.size <= 8 else np.linalg.norm(arr, axis=0)[:8])
for name in ["fit117", "fit118"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys", list(obj))
        print("report:", str(obj.get("report"))[:1400])

print("\n=== RECENT HELPER DEFINITIONS ===")
for name in ["eval_hpre128", "local_core_step119", "core_step119",
             "recurrent_step119", "prelude_features111"]:
    fn = globals().get(name)
    if callable(fn):
        print(name, inspect.signature(fn))
        try:
            print(inspect.getsource(fn)[:2500])
        except (OSError, TypeError):
            print("names:", fn.__code__.co_names,
                  "constants:", str(fn.__code__.co_consts)[:1400])

print("\n=== RECENT VARIABLE NAMES ===")
print(" ".join(sorted(k for k in globals()
                     if re.search(r"12[5-8]", k))))
print("\n=== HELPER NAMES ===")
print(" ".join(sorted(k for k, v in list(globals().items())
                     if callable(v) and getattr(v, "__module__", "") == "__main__"))[:1800])
print("\n=== RECURRENT INPUT TENSORS ===")
for name in NAMES:
    if "b_in" in name or "delta_in" in name:
        print(name, SHAPES[name])
```