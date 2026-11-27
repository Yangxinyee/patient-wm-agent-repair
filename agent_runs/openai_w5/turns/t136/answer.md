The earlier checks found similar effects across the two glucose sensors and the two MAP sensors, though persistence estimates still depend on feedback assumptions. I’ll now inspect the candidate’s weight layout and reserved coordinates. A usable channel must carry rates without materially changing the model’s ordinary action representation.

```python
import inspect
import dis
import io

ns120 = dict(globals())
print("=== CHANNEL SETTINGS ===")
for n, o in sorted(ns120.items()):
    if re.search(r"(117|118)$", n):
        if isinstance(o, (int, float, str)):
            print(n, repr(o)[:120])
        elif isinstance(o, np.ndarray) and o.size <= 12:
            print(n, o)
        elif isinstance(o, torch.Tensor) and o.numel() <= 12:
            print(n, o)

print("\n=== BEST NORMALIZED FIT COMPONENTS ===")
for j, o in enumerate(best_norm118):
    if isinstance(o, (np.ndarray, torch.Tensor)):
        print(j, type(o).__name__, tuple(o.shape))
    elif isinstance(o, dict):
        print(j, o)
    else:
        print(j, type(o).__name__, repr(o)[:180])

print("\n=== ENCODER CANDIDATE CHANGES ===")
for n, o in enc_candidate118.items():
    x = np.asarray(o.detach().cpu() if isinstance(o, torch.Tensor) else o)
    original = np.asarray(load_original(n))
    print(n, x.shape,
          "delta_rms", round(float(np.sqrt(np.mean((x-original)**2))), 6),
          "delta_max", round(float(np.max(np.abs(x-original))), 5))
print("reserved:", R117)
print("candidate norm gamma:", np.asarray(enc_candidate118["encode_action.norm.weight"])[R117])
print("candidate norm bias:", np.asarray(enc_candidate118["encode_action.norm.bias"])[R117])

print("\n=== RELEVANT ATTENTION SHAPES ===")
for n in NAMES:
    if n.startswith("core.prelude.0.attn"):
        print(n, SHAPES[n])

print("\n=== RECENT HELPER IMPLEMENTATIONS ===")
for n in ["enc_set117", "channel_raw_target117", "channel_dataset117"]:
    f = ns120[n]
    print("\n", n, inspect.signature(f))
    try:
        text = inspect.getsource(f)
    except (OSError, TypeError):
        buf = io.StringIO()
        dis.dis(f, file=buf)
        text = buf.getvalue()
    print(text[:1800])
```
