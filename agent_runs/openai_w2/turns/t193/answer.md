The validated edit is intact. The earlier analysis also caught a recording-policy bias in the initial insulin estimates, so those crude estimates should not drive a stronger edit. I’ll inspect the corrected effect estimates and the decoder trade-offs. A better norepinephrine decoder should retain dose-edit sensitivity while producing less signal on factual records.

```python
import inspect
import re

print("=== CONSTRAINED DECODER DIAGNOSTICS ===")
print(repr(reports147)[:3500])
for name169 in [
    "Hfact147", "Hprobe147", "C147", "F147", "X147", "Y147",
    "Dparts147", "Yparts147", "mu147", "sd147", "delta_scale150"
]:
    obj169 = globals().get(name169)
    print(name169, type(obj169).__name__, getattr(obj169, "shape", None))
    if isinstance(obj169, dict):
        print({k169: getattr(v169, "shape", type(v169).__name__)
               for k169, v169 in obj169.items()})
for name169 in ["report_decoder147", "fold_decoder147"]:
    fn169 = globals().get(name169)
    if callable(fn169):
        print(name169, inspect.signature(fn169))
        print("Dependencies:", fn169.__code__.co_names)

print("\n=== CORRECTED IDENTIFICATION AND PERSISTENCE NOTES ===")
sections169 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(),
                       flags=re.MULTILINE)
for prefix169 in [
    "## Run 6:", "## Run 10:", "## Run 11:", "## Run 12:",
    "## Runs 157"
]:
    matches169 = [s169 for s169 in sections169 if s169.startswith(prefix169)]
    for s169 in matches169:
        print(s169[:1100])

with open("/work/notes.md", "a") as f169:
    f169.write(
        "\n## Restart recovery, runs 167–169\n"
        "Verified that only core.b_in.weight is edited and it exactly matches "
        "validated_B151, initial-safe decoder with gains [-10,+12]. "
        "Exact reached-hour comparison from run166: per-stay insulin x2/half "
        "-12.1599/+5.4133 versus assumption-dependent record targets "
        "-12.8893/+6.4447; NE x2/half +4.2690/-2.0001 versus "
        "+7.0881/-3.5440. Pooled-hour averages differ from per-stay averages; "
        "use the evaluation's per-stay aggregation for final interpretation. "
        "No weight changes during recovery. Inspecting decoder efficiency "
        "before any stronger NE candidate.\n"
    )
```
