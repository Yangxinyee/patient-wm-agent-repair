The repeated output confirms restoration but adds no new diagnostic evidence. I’ll now inspect the exact compensation and latent-carrier formulas. If compensation cancels or distorts dose changes, a joint factual-and-dose-contrast regression should expose that problem.

```python
import inspect
import dis
import io
import re
from pathlib import Path

text253 = Path("/work/notes.md").read_text()
print("=== DESIGN NOTES ===")
budget253 = 3200
for entry253 in text253.split("\n\n"):
    match253 = re.search(r"Python runs?:?\s+(\d+)", entry253)
    if match253 and 185 <= int(match253.group(1)) <= 206:
        out253 = entry253[:650]
        if len(out253) <= budget253:
            print(out253)
            budget253 -= len(out253)

for name253, limit253 in (
    ("latent206", 2300),
    ("hidden220", 1800),
    ("prelude_parts145", 1800),
):
    f253 = globals()[name253]
    print("\nHELPER:", name253, inspect.signature(f253))
    print("references:", f253.__code__.co_names)
    try:
        source253 = inspect.getsource(f253)
    except (OSError, TypeError):
        instructions253 = dis.get_instructions(f253)
        source253 = "\n".join(
            f"{op.opname} {op.argrepr}" for op in instructions253
        )
    print(source253[:limit253])

print("=== CALIBRATION CACHE ===")
cal253 = []
for name253, value253 in list(globals().items()):
    if re.search(r"(234|235|242|243|250)$", name253):
        if hasattr(value253, "shape"):
            cal253.append((name253, tuple(value253.shape)))
        elif isinstance(value253, dict):
            cal253.append((name253, "dict", list(value253)[:4]))
print(cal253[:30])

with Path("/work/notes.md").open("a") as log253:
    log253.write(
        "\nPython run 253; forwards 20. Checkpoint222 remains restored. "
        "Inspecting exact latent, attention, and FFN helpers plus calibration "
        "cache to diagnose coupled compensation. No weights changed.\n"
    )
```