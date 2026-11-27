The local calculation predicts zero drug signals without treatment, but the GPU states disagree. I’ll check the exact computation and state layout before editing further. An input, activation, or indexing mismatch would explain this contradiction.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== RECENT FORWARD FILES ===")
for k in (23, 24, 25):
    p = Path(f"/work/fwd_{k}")
    if p.exists():
        print(k, [(q.name, q.stat().st_size) for q in p.iterdir()])
print("=== LATENT TENSOR SHAPES ===")
print({n: SHAPES[n] for n in NAMES if n.startswith(("z_post", "z_in"))})
print("=== HELPER IMPLEMENTATION DETAILS ===")
for name in ("mlp136", "posterior137", "linear136", "norm136", "core_step136"):
    f = globals().get(name)
    if callable(f):
        print(name, inspect.signature(f))
        print("names:", f.__code__.co_names)
        print("constants:", f.__code__.co_consts)
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 265 — resolve local/GPU neutrality contradiction\n"
            "- Neutral209 remains installed; no new weights changed.\n"
            "- Reading the exact architecture and state layout before relying on local private-channel calculations.\n")
```