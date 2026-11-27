The last result was a PCA of the action token `u` over the 10,032 drug bundles. Nearly all of its variation lies along one direction (first eigenvalue 0.0247, then 0.0011, then below 1e-4). So the dose enters the core as an essentially one-dimensional signal. That supports Stage 1 of my plan: let the unused z channel carry a copy of `u` plus the dose residual (recorded dose minus the dose the policy would predict from context).

Before building that edit I need two details I lost at the restart:
- **Prelude tokens:** exactly how the z token is built (z_in, type embeddings).
- **Core recurrence:** how `h` is updated, i.e. whether any direction can persist for many hours.

The true insulin effect builds up and fades over about 10 hours, and the model currently forgets dose information within about an hour. So whether a slow-decaying channel exists decides whether I can match the time course, or only calibrate a one-step gain.

How I'll read the result:
- **Persistent channel possible:** the core update has a residual or gated path for `h` that the edit could use to hold a decaying dose memory.
- **Calibration only:** `h` is fully recomputed from memory each step. Then the shape is fixed by the model's own glucose dynamics, and I can only tune the one-step effect size against the readouts using my offline replica.

This run only reads files and lists my replica's functions.

```python
import re, os
arch = open('/model/ARCHITECTURE.md').read()
print(len(arch))
print(arch[:14000])
print("=====")
for f in ['/work/wm2.py','/work/wm2_rollout.py','/work/wm.py','/work/setup.py']:
    if os.path.exists(f):
        src = open(f).read()
        print(f, len(src), [m for m in re.findall(r'^def (\w+)\(', src, flags=re.M)])
print([k for k in list(globals().keys()) if not k.startswith('_')][:200])
```