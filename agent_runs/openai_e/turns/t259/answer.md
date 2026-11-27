The unused sample failed the limit: NLL increased by 0.00567. That makes the current trial too fragile to accept unchanged. I’ll inspect the detector fits and output-head structure for a way to reduce factual leakage while preserving dose sensitivity. If that separation is not possible, I’ll reduce the edit’s strength.

```python
print("=== INSULIN CONSTRUCTION HISTORY ===")
for lo, hi in ((20600, 21600), (23174, 24700), (26000, 28200), (38688, 39240)):
    print(notes221[lo:hi])

print("\n=== EDIT MATRIX AND FACTOR AUDIT ===")
for fn in ("baseline77.npz", "insulin_candidate58.npz", "insulin_candidate63.npz",
           "insulin_trial219.npz"):
    a = np.load("/work/" + fn)
    print(fn)
    for k in a.files:
        x = a[k]
        if x.ndim == 0:
            print(" ", k, float(x))
        else:
            print(" ", k, x.shape, "norm", round(float(np.linalg.norm(x)), 5))
    if "delta" in a.files:
        sv = np.linalg.svd(a["delta"], compute_uv=False)
        print(" singular values", sv[:3])

print("\n=== VALUE-HEAD TENSORS ===")
for n in NAMES:
    if any(x in n for x in ("attr_heads.value", "embed_transition.value", "last_numeric")):
        print(n, SHAPES[n])

print("\n=== COMPUTATION HELPERS ===")
for k in list(globals()):
    v = globals()[k]
    if callable(v) and re.search(r"(?:prelude|core_|coda|decode|build_static|embed|counter|pair|prepare|head)", k):
        try:
            print(k, inspect.signature(v))
        except Exception:
            pass

print("\n=== RECENT TRAINING FIT OBJECTS ===")
ks223 = [k for k in list(globals()) if re.search(r"(?:18[5-9]|19[0-8])$", k)]
for k in ks223[:65]:
    v = globals()[k]
    if hasattr(v, "shape"):
        print(k, v.shape)
    elif callable(v):
        print(k, inspect.signature(v))
    elif isinstance(v, dict):
        print(k, "dict", list(v)[:5])
    elif isinstance(v, (int, float, str)):
        print(k, repr(v)[:100])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 223: fresh factual sample rejects narrow trial as robust choice\n"
            "Forward26 NLL 1.63610 versus original 1.63043, cost .00567 on 100 unused tuning stays. "
            "Trial219 therefore lacks robust factual margin. Investigating closed-form leakage reduction "
            "or output-head distillation with causal-direction guards; no new weights installed. "
            "Python runs: 223; forwards: 26.\n")
```