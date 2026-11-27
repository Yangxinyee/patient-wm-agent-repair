The record analysis supports a delayed, declining insulin response and a faster-decaying norepinephrine response; a simple one-step correction would miss that distinction. The retained probe can test how posterior changes reach the next readings. I’ll inspect its results and the weight-loading functions before editing, so that the first candidate is based on a verified model response rather than an assumed latent direction.

```python
print("=== LATE EDIT / LATENT NOTES ===")
for section108 in notes105.split("\n## "):
    match108 = re.match(r"Run(?:s)?\s+(\d+)", section108)
    if match108 and 85 <= int(match108.group(1)) <= 97:
        print("##", section108[:380])

print("\n=== ESTIMATED RECORD KERNELS ===")
for sensor108 in kernels19:
    print(sensor108, "kernel", np.round(kernels19[sensor108], 4),
          "SE", np.round(kernel_se19[sensor108], 4))
print("AR2 predictions:", ar2_predictions19)

print("\n=== WEIGHT ACCESS ===")
for fun108 in (w63, linear63, mlp63, context64):
    print("\n", fun108.__name__)
    show91(fun108, limit=1150)

print("\n=== LATENT DIRECTIONS ===")
for name108, value108 in directions93.items():
    print(name108, "shape", np.shape(value108))
    if isinstance(value108, dict):
        print({k: (tuple(v.shape) if hasattr(v, "shape") else v)
               for k, v in value108.items()})
    else:
        a108 = np.asarray(value108)
        print("norm", np.linalg.norm(a108), "range", a108.min(), a108.max())

print("\n=== PROBE VARIABLES ===")
for key108, val108 in list(globals().items()):
    if key108.endswith(("88", "89", "90", "91", "92", "93", "94")):
        if isinstance(val108, dict):
            print(key108, "dict keys", list(val108)[:12])
        elif hasattr(val108, "shape") and (
            val108.ndim <= 2 if hasattr(val108, "ndim") else True):
            print(key108, tuple(val108.shape))
```
