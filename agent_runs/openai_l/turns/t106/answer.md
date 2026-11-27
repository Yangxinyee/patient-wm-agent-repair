The last fits preserve factual accuracy only when the dose effects remain too weak or have the wrong sign. That argues against a value-head-only repair. I’ll recover the upstream dose encoding and the record-based causal evidence next. An upstream correction would be promising only if it strengthens the supported dose responses without exceeding the factual-loss limit.

```python
from pathlib import Path
import inspect, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes98 = Path("/work/notes.md").read_text()
sections98 = re.split(r"(?=^## )", notes98, flags=re.M)
evidence98 = [
    s for s in sections98
    if re.search(r"DML|discontinuity|randomiz|instrument|causal|kernel|threshold|policy", s, re.I)
    and not re.search(r"^## (?:Run|Runs) (?:6[9]|[7-9]\d)", s)
]
print("\n=== EARLIER CAUSAL EVIDENCE ===")
print("\n".join(evidence98)[:16000])

snapshot98 = dict(globals())
print("\n=== RETAINED FUNCTION INTERFACES ===")
for name98, obj98 in sorted(snapshot98.items()):
    if inspect.isfunction(obj98) and getattr(obj98, "__module__", "") == "__main__":
        try:
            print(name98, inspect.signature(obj98))
        except Exception:
            print(name98)

print("\n=== RELEVANT WEIGHT SHAPES ===")
for name98 in NAMES:
    if re.search(r"atom|dose|value|embed|posterior|prior|transition|slot|decoder|head|input", name98, re.I):
        print(name98, SHAPES[name98])

print("\n=== RETAINED FIT / CAUSAL ARRAYS ===")
for name98, obj98 in sorted(snapshot98.items()):
    if re.search(r"caus|kernel|target|rd_|dml|coef|fit|cf|features|heads|x97|y97|x96", name98, re.I):
        shape98 = getattr(obj98, "shape", None)
        if shape98 is not None:
            print(name98, type(obj98).__name__, tuple(shape98))
        elif isinstance(obj98, dict):
            print(name98, "dict keys:", list(obj98)[:16])

with Path("/work/notes.md").open("a") as f98:
    f98.write(
        "\n## Run 98 / upstream repair recovery\n"
        "Run 97 confirms that a factual-preserving raw-head correction cannot recover "
        "the supported drug slopes. Recovering architecture, causal evidence, and retained "
        "interfaces before testing an upstream dose-residual correction. No weights edited.\n"
    )
```
