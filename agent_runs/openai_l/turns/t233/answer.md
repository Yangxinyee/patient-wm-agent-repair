The caches are intact; inspection stopped because one cache uses tuples rather than dictionaries. No weights changed. I’ll check that structure and the fitting helpers so the next experiment uses correctly paired factual and edited-dose inputs.

```python
def describe210(obj, depth=0):
    if isinstance(obj, np.ndarray):
        return ("numpy", obj.shape, str(obj.dtype))
    if torch.is_tensor(obj):
        return ("torch", tuple(obj.shape), str(obj.dtype))
    if isinstance(obj, dict):
        return {str(k): describe210(v, depth + 1) for k, v in list(obj.items())[:12]}
    if isinstance(obj, (tuple, list)) and depth < 2:
        return [describe210(v, depth + 1) for v in obj[:12]]
    return (type(obj).__name__, str(obj)[:140])

for name, obj in [
    ("canonical x2", canon_cases200[("insulin", 2.0)]),
    ("paired x2", cases207[("insulin", 2.0)]),
    ("dose direction", directions207[("insulin", 0.1)]),
    ("family masks", family204),
]:
    print(name, describe210(obj))

print("\nFIT HELPERS:")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and any(s in name.lower() for s in [
        "ridge", "decoder", "head", "collect", "nll", "fit", "metrics", "gelu"
    ]):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

print("\nROW/SPLIT VARIABLES:")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in ["rows179", "rows178", "meta179", "meta178", "train179",
                                     "valid179", "mask179", "tr179", "va179", "tf179", "vf179",
                                     "train200", "valid200", "tr200", "va200", "row81", "meta81"]):
        print(name, describe210(globals()[name]))
print("\nOTHER 179 VARIABLES:", [n for n in sorted(globals()) if "179" in n])
print("\nGATE RANGE:",
      np.quantile(Xraw204 @ gatecanon200, [0, .01, .5, .99, 1]))
print("\nC179 FIRST TWO ROWS:", C179[:2])
print("B179 FIRST TEN:", B179[:10])
print("\nNOTEBOOK CAUSAL-EVIDENCE HEADINGS:")
notes210 = Path("/work/notes.md").read_text()
for line in notes210.splitlines():
    if line.startswith("##") or ("kernel" in line.lower() and len(line) < 420):
        print(line[:420])
```
