The recovered notes support negative insulin effects and positive norepinephrine effects, but explicitly reject the conflicting state-space estimates. The output was truncated before the controller details. I’ll retrieve those interfaces next; their dimensions and perturbation responses must agree before any further edit.

```python
print("=== CONTROLLER COMPONENTS ===")
for n in ["P125", "T125", "QE125", "SP125", "Vnew125", "Wrepair125",
          "chosen125", "R118", "Vsignal", "route122", "Jselected"]:
    v = globals().get(n, None)
    if isinstance(v, (np.ndarray, torch.Tensor)):
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        print(n, tuple(a.shape), "norm", float(np.linalg.norm(a)),
              "range", (float(a.min()), float(a.max())))
        if a.size <= 36:
            print(a)
    else:
        print(n, type(v).__name__, str(v)[:900])
print("=== FUNCTIONS ===")
for n, v in list(globals().items()):
    if not n.startswith("_") and inspect.isfunction(v):
        try:
            print(n, inspect.signature(v))
        except Exception:
            pass
print("=== RELATED NAMES ===")
related = [n for n in globals() if not n.startswith("_") and
           any(s in n.lower() for s in ["125", "118", "122", "cache", "kernel",
                                       "causal", "source", "src", "code", "state",
                                       "prelude", "wire", "controller"])]
print(related)
print("=== EVIDENCE AFTER RUN 9 ===")
print(_notes139[7000:10500])
print("=== RECENT NOTEBOOK SECTIONS ===")
print("\n".join(x for x in _notes139.splitlines() if x.startswith("#"))[-1800:])
```
