The records support lower glucose after insulin and higher MAP after norepinephrine, but those associations alone do not establish the multi-hour effects. I will restore the factual-safe baseline and locate the cached state and head calculations. They should let us test whether dose information disappears in the state update or remains available but is underused by the reading head.

```python
candidate255 = {n: np.asarray(load(n), dtype=np.float32).copy() for n in edited()}
assert "safe236" in globals()
for n in list(edited()):
    drop_edit(n)
for n, v in safe236.items():
    save_edit(n, np.asarray(v, dtype=np.float32))
print("Restored safe236:", list(edited()))

print("\n=== FUNCTION NAMES ===")
_fn255 = [k for k, v in list(globals().items())
          if callable(v) and getattr(v, "__module__", None) == "__main__"
          and not k.startswith("_")]
print(", ".join(_fn255))

print("\n=== RELEVANT EARLIER NOTES ===")
_hit255 = [l for l in open("/work/notes.md").read().splitlines()
           if any(s in l.lower() for s in
                  ["distributed", "lag", "head edit", "value head", "value-head",
                   "decoder", "direct ", "jacobian", "dose response", "dose-response"])]
for l in _hit255[:25]:
    print(l[:170])
print("--- later ---")
for l in _hit255[-16:]:
    print(l[:170])

print("\n=== STATE / HEAD CACHE NAMES ===")
_cache255 = []
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if any(s in k.lower() for s in ["cache", "jac", "with_code", "readout", "head_input"]):
        if isinstance(v, dict):
            _cache255.append((k, "dict", list(v.keys())[:8]))
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            _cache255.append((k, tuple(v.shape)))
for item in _cache255[-25:]:
    print(item)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 255\n"
            "Preserved quarter-strength candidate255 and restored safe236 because forward27 "
            "showed no useful effect gain. Recovering state/head functions and earlier unsuccessful "
            "head routes to distinguish lost state signal from weak decoding. "
            "Counts: 255 Python runs, 27 forwards.\n")
```