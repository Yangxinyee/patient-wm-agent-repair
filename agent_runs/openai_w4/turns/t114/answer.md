The perturbation reaches the posterior, but adding 5 to a latent coordinate barely changes `out`. That supports a nearly inactive latent-to-state pathway, rather than a probe wiring error. I’ll inspect the transition scales and the gated insulin-innovation decoder before designing a stronger connection. The repair should respond to changed doses while remaining quiet on ordinary records.

```python
import numpy as np, torch, dis

print("=== TRANSITION SCALES ===")
for k, v in capture81.items():
    if hasattr(v, "shape") and k not in ("mem", "ages", "mask", "agevec", "indices"):
        a = np.asarray(v)
        print(k, a.shape,
              "rms", round(float(np.sqrt(np.mean(a*a))), 6),
              "maxabs", round(float(np.max(np.abs(a))), 6))

print("\n=== INSULIN INNOVATION CHANNEL ===")
print("gated decoder dependencies:", gated_innovation94.__code__.co_names)
print("gated decoder constants:", gated_innovation94.__code__.co_consts)
dis.dis(gated_innovation94, depth=0)

print("\n=== CANDIDATE CONSTRUCTION OBJECTS ===")
for n, v in sorted(list(globals().items())):
    if n.startswith("_") or not n.endswith("94"):
        continue
    if hasattr(v, "shape"):
        print(n, tuple(v.shape))
    elif isinstance(v, (list, tuple)):
        print(n, repr(v)[:180])
    elif isinstance(v, dict):
        print(n, "keys", list(v.keys())[:10])
    elif not callable(v):
        print(n, repr(v)[:180])

print("\n=== PRELUDE WEIGHT NAMES ===")
for n in NAMES:
    if "core.prelude.0" in n or "z_post" in n:
        print(n, SHAPES[n])

print("\n=== EARLY RECORD-BASED EVIDENCE ===")
print(open("/work/notes.md").read()[3500:6800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 101\nVerified posterior injection reaches z but changes out only about 1e-6 RMS for a +5 latent perturbation. The latent-to-state route is nearly inactive. Inspecting prelude/state scales and the retained gated insulin innovation before constructing a selective stronger connection. No edits.\n")
```
