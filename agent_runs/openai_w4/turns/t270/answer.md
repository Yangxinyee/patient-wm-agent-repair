The architecture confirms that dose information can affect both latent inference and persistent recurrent state, so an immediate-response fit alone cannot validate these rollouts. The inspection was truncated before the retained estimates appeared. I’ll retrieve those estimates and construction helpers compactly; agreement between independent record-based estimates would strengthen the calibration targets.

```python
import inspect, re
from pathlib import Path

g235 = dict(globals())
print("=== RELEVANT OBJECT NAMES ===")
pat235 = re.compile(r"kernel|target|caus|ridge|repair|connection|backup|channel|innovation|gate|impulse|effect")
items235 = []
for name, value in sorted(g235.items()):
    if pat235.search(name.lower()) and not inspect.isfunction(value):
        desc = type(value).__name__
        if hasattr(value, "shape"):
            desc += str(tuple(value.shape))
        elif isinstance(value, dict):
            desc += ":" + ",".join(map(str, list(value.keys())[:10]))
        elif isinstance(value, (list, tuple)):
            desc += ":" + str(len(value))
        items235.append(name + "=" + desc)
print("\n".join(items235)[:3500])

print("=== REMAINING FUNCTION INTERFACES ===")
for name, value in sorted(g235.items()):
    if name >= "s" and inspect.isfunction(value) and value.__module__ == "__main__":
        try:
            print(name, inspect.signature(value))
        except Exception:
            print(name)

print("=== RECORD-IDENTIFICATION NOTE EXCERPTS ===")
lines235 = Path("/work/notes.md").read_text().splitlines()
hits235 = [(i, line) for i, line in enumerate(lines235)
           if re.search(r"discontin|instrument|cutoff|identifi|causal|kernel|target|per.unit",
                        line, re.I)]
selected235 = hits235[:25] + hits235[-17:]
text235 = "\n".join(f"{i}: {line}" for i, line in selected235)
print(text235[:3400])
print("Notebook lines:", len(lines235))

with open("/work/notes.md", "a") as f:
    f.write("""
## Run 235
Architecture recovered; previous display truncated before retained objects.
Retrieving compact kernel/target/construction metadata. Current saved
threshold-26 candidate unchanged; GPU factual delta .00389.
""")
```
