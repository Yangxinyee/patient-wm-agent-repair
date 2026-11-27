The earlier analysis gives provisional tuning-set rollout targets of about −11.9 and +5.9 for insulin doubling and halving, and +4.5 and −2.2 for norepinephrine. These are estimates, not known truth. I’ll now inspect the head reconstruction and private-channel settings so the next candidate can be tested against those targets without confusing dose transport with causal calibration.

```python
import inspect, re, types, torch

print("=== CALIBRATION OBJECT NAMES ===")
rx144 = re.compile(
    r"14[01]|proxy|tilt|distill|reserv|neuron|slope|best|clean|channel|coef|target|R117",
    re.I
)
matches144 = [
    n for n, o in list(globals().items())
    if not n.startswith("__") and rx144.search(n) and not isinstance(o, types.ModuleType)
]
print(" ".join(matches144)[:2500])

print("=== SMALL CALIBRATION OBJECTS ===")
for n in matches144:
    o = globals()[n]
    if isinstance(o, dict) and len(o) < 30:
        print(n, "keys:", list(o)[:20])
    elif hasattr(o, "shape") and np.prod(o.shape) <= 24:
        print(n, np.asarray(o))
    elif isinstance(o, (float, int, str)) and len(str(o)) < 160:
        print(n, o)

print("=== HEAD FEATURE RECONSTRUCTION ===")
for fname144, limit144 in [
    ("head_features_layout", 3200),
    ("reconstruct_head", 3500),
]:
    fn144 = globals()[fname144]
    try:
        text144 = inspect.getsource(fn144)
        print(fname144, "\n", text144[:limit144])
        print("Full source characters:", len(text144))
    except Exception as exc144:
        print(fname144, "source unavailable:", repr(exc144))
        print("Referenced names:", fn144.__code__.co_names)

print("=== PRIVATE-CHANNEL SETTINGS ===")
print("Reserved state coordinates:", R117)
print("Selected decay rates:", load("core.flow.a")[R117])
print("Selected out-norm scales:", load("core.out_norm.weight")[R117])
print("Head tensor shapes:", {
    n: SHAPES[n] for n in NAMES if n.startswith("attr_heads.value_bin")
})
print("Working-weight dictionaries:", [
    (n, len(o)) for n, o in list(globals().items())
    if isinstance(o, dict) and "attr_heads.value_bin.0.weight" in o
][:20])
```
