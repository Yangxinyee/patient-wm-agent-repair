The large query mismatch means the Monte Carlo audit is still unreliable; it does not establish that the rejected edit was safe or unsafe. I’ll recheck the exact architecture and reconstruction code. Agreement with the original GPU predictions—not merely local edit equivalence—will be the required test.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())
print("\n=== LIVE FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, sig)

print("\n=== LIVE ARRAYS / MODEL OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_"):
        continue
    shape = getattr(v, "shape", None)
    if shape is not None:
        print(k, type(v).__name__, tuple(shape))
    elif isinstance(v, dict) and any(s in k.lower() for s in
                                    ("weight", "state", "head", "cache", "cand", "query", "voc")):
        print(k, "dict", len(v), "keys:", list(v)[:12])

print("\n=== ACTIVE EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 140: restart audit\n"
            "- No active edits. Run 139's query mismatch invalidates its Monte Carlo comparison.\n"
            "- Re-reading the exact architecture and live reconstruction functions before another candidate.\n"
            "- Require original CPU/GPU agreement under the correct latent/query distribution.\n")
```