The recovered notes support an immediate insulin effect near −2 per rate unit, including independent threshold evidence, and a norepinephrine effect near +3.1. The earlier state-space estimates had conflicting signs and were rejected. I’ll now inspect the live feature helpers so the correction uses the accepted evidence; it must preserve factual logits while improving dose contrasts.
```python
import inspect, re
from pathlib import Path

print("FUNCTIONS:")
for n175, o175 in sorted(list(globals().items())):
    if inspect.isfunction(o175):
        try:
            print(n175, inspect.signature(o175))
        except Exception:
            print(n175)

print("\nRECENT VARIABLE INTERFACES:")
for n175, o175 in sorted(list(globals().items())):
    if re.search(r"(167|168|169|170|171|172|173)$", n175):
        if isinstance(o175, dict):
            print(n175, "dict", [
                (str(k), tuple(v.shape) if hasattr(v, "shape") else
                 ("dict:" + ",".join(map(str, list(v.keys())[:12])) if isinstance(v, dict)
                  else repr(v)[:100]))
                for k, v in list(o175.items())[:18]])
        elif hasattr(o175, "shape"):
            print(n175, tuple(o175.shape))
        else:
            print(n175, type(o175).__name__, repr(o175)[:160])

print("\nVALUE-HEAD TENSORS:")
for n175 in NAMES:
    if n175.startswith("attr_heads.value_bin") or n175.startswith("last_numeric"):
        print(n175, SHAPES[n175])

notes175 = Path("/work/notes.md").read_text()
parts175 = notes175.split("\n## ")
hits175 = [p for p in parts175
           if any(s in p.lower() for s in ("eight-lag", "8-lag", "distributed", "rd estimate"))
           and not p.startswith(("Run 12", "Run 13", "Run 14", "Run 15", "Run 16", "Run 17"))]
print("\nKERNEL-EVIDENCE NOTES:")
print("\n\n".join(hits175[-5:])[:3000])
```