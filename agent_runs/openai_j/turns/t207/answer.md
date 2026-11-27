The last run confirmed that the local evaluator works; it did not establish any additional causal improvement. I’ll recover the architecture and record-derived evidence before extending the MAP edit. A stronger response is useful only if the evidence supports it and factual loss remains within budget.

```python
from pathlib import Path
import re

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION")
print(Path("/data/README.md").read_text())

notes186 = Path("/work/notes.md").read_text()
sections186 = re.split(r"(?=^## )", notes186, flags=re.M)
selected186 = [
    s for s in sections186
    if re.search(r"causal|identif|instrument|IV\b|kernel|evidence|dynamic",
                 s.split("\n", 1)[0], re.I)
]
print("\nCAUSAL NOTEBOOK SECTIONS")
for s in (selected186[:5] + selected186[-7:]):
    print(s[:1800])

print("\nAVAILABLE LOCAL FUNCTIONS")
print(sorted(
    k for k, v in list(globals().items())
    if callable(v) and getattr(v, "__module__", "") == "__main__"
))

print("\nRELEVANT VARIABLE INVENTORY")
for k, v in list(globals().items()):
    if not re.match(r"^(map16|map17|head177|wm_|mc169|iv|dyn|sensor|best|tune|train)", k):
        continue
    if isinstance(v, dict):
        print(k, "dict", list(v.keys())[:24])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v), repr(v[:8])[:220])
    elif isinstance(v, (int, float, str)):
        print(k, repr(v)[:220])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart audit, run 186\n"
            "- Run 185 confirmed the local stochastic head evaluator returns per-reading NLL and means.\n"
            "- Existing combined four-tensor edit remains active; no new edits made.\n"
            "- Recovering architecture, causal evidence, and candidate definitions before further MAP changes.\n")
```
