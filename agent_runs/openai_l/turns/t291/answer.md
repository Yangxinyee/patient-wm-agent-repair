The retained evidence supports a delayed insulin response and a faster-decaying norepinephrine response, with agreement across sensors. The current head uses gated features for both drugs, but their fitted corrections may still interfere. I’ll inspect the regression matrices and factual-validation caches; this will show whether we can test that interference without changing the upstream model.

```python
print("=== RECORD IMPULSE RESPONSES ===")
for n258, v258 in kernels19.items():
    print(n258, v258)

print("\n=== RECONSTRUCTED CURRENT FEATURE ARCHITECTURE ===")
arch258 = make_arch215("pair4", 0.1)
print("type:", type(arch258).__name__)
if isinstance(arch258, dict):
    for k258, v258 in arch258.items():
        if isinstance(v258, np.ndarray):
            print(k258, v258.shape, v258.dtype)
        elif isinstance(v258, dict):
            print(k258, "dict", list(v258)[:12])
            for kk258, vv258 in list(v258.items())[:1]:
                print(" example", kk258,
                      {q: x.shape if isinstance(x, np.ndarray) else type(x).__name__
                       for q, x in vv258.items()} if isinstance(vv258, dict)
                      else (vv258.shape if isinstance(vv258, np.ndarray) else type(vv258).__name__))
        else:
            print(k258, str(v258)[:150])

print("\n=== RECENT LIVE REGRESSION AND AUDIT OBJECTS ===")
for n258, v258 in sorted(list(globals().items())):
    if re.search(r"(228|231|232|233|234|235|237|239|240|241|242|243|244|245|246|249|250|251|252|253|254)$", n258):
        if isinstance(v258, np.ndarray):
            print(n258, v258.shape, v258.dtype)
        elif isinstance(v258, dict):
            print(n258, "dict", list(v258)[:12])
        elif hasattr(v258, "shape"):
            print(n258, type(v258).__name__, v258.shape)
        elif not callable(v258):
            print(n258, type(v258).__name__, str(v258)[:100])

print("\n=== SCORE FUNCTION ===")
import dis
for ins258 in dis.get_instructions(score234):
    print(ins258.opname, ins258.argrepr)
```
