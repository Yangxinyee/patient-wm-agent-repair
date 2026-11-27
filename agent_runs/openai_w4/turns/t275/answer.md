The architecture confirms that generated readings feed back into later predictions, so matching a one-step drug response is insufficient. The inspection was truncated before the estimation details. I will retrieve only the identification summary and repair-function names next; this should distinguish a units error from a genuine rollout-calibration gap.

```python
import re
from pathlib import Path

r240_g = dict(globals())
print("=== CUSTOM FUNCTION NAMES ===")
r240_funcs = [
    n for n, v in r240_g.items()
    if callable(v) and getattr(v, "__module__", None) == "__main__"
    and not n.startswith("__")
]
print(", ".join(sorted(r240_funcs)))

print("\n=== KERNEL / POLICY OBJECT NAMES ===")
for token in ("kernel", "policy", "direction", "gain"):
    names = [n for n in r240_g if token in n.lower() and not n.startswith("r239")]
    print(token + ": " + ", ".join(sorted(names)))

print("\n=== INITIAL IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:4300])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 240\n"
            "Run-239 architecture confirms recurrent and generated-reading feedback. "
            "Using bounded identification/interface inspection after truncated output; "
            "no weights changed.\n")
```