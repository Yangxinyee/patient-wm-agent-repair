The stronger norepinephrine edit failed the factual-fit limit, so the smaller checkpoint was restored. I’ll recover the effect estimates and checkpoint details before making another change; any new candidate needs broader factual validation.

```python
from pathlib import Path
import re, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes_text[:6500])
print("\n=== CAUSAL / TARGET NOTEBOOK SECTIONS ===")
sections = re.split(r"(?=^## )", notes_text, flags=re.M)
relevant = [
    s for s in sections
    if re.search(
        r"identification|instrumental|causal target|record-derived target|"
        r"long.horizon|provisional target|true effects|target effects",
        s, flags=re.I
    )
]
for s in relevant[-12:]:
    print(s[:1800])

print("\n=== KERNEL FUNCTION NAMES ===")
fn_names = sorted(
    k for k, v in list(globals().items())
    if inspect.isfunction(v) and not k.startswith("_")
)
print(fn_names)

print("\n=== CHECKPOINT / CANDIDATE INVENTORY ===")
for k, v in sorted(list(globals().items())):
    if re.match(r"^(r235|r240|r242|r244|r246|r247|c240|c242|c244|c246|c247)", k):
        if isinstance(v, dict):
            print(k, {
                str(kk): (getattr(vv, "shape", None) or type(vv).__name__)
                for kk, vv in v.items()
            })
        elif hasattr(v, "shape"):
            print(k, "shape", v.shape)
        elif inspect.isfunction(v):
            print(k, "args", inspect.getfullargspec(v).args)
        elif isinstance(v, (str, int, float, bool, list, tuple)):
            print(k, repr(v)[:500])
        else:
            print(k, type(v).__name__)

print("\n=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 249: restart recovery\n"
        "Retained the restored NE .50 cancellation checkpoint. Recovering "
        "architecture, record-derived causal targets, and exact candidate "
        "interfaces before broader factual validation. No weights changed.\n"
    )
```
