The notebook’s early effect estimates disagreed, and it explicitly rejected the provisional state-space targets. I’ll recover the later records-based evidence and live candidate objects. The targets should rest on those identification checks, not on the model’s own rollouts.

```python
import re

print("=== LATER RECORD EVIDENCE ===")
start250 = notes249.find("## Run 9")
print(notes249[start250:start250 + 4300])

print("=== LIVE FUNCTION NAMES ===")
print(sorted(k for k, v in list(globals().items())
             if inspect.isfunction(v) and not k.startswith("_")))

print("=== CURRENT CANDIDATE OBJECTS ===")
for k in ["grid242", "T235", "backup248", "backup237", "kernels19",
          "ids249", "staycol249"]:
    if k in globals():
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys()) if isinstance(v, dict) else
              (repr(v)[:700] if k != "ids249" else ""))

print("=== RECENT CACHE NAMES ===")
print([k for k in sorted(globals())
       if re.search(r"(228|234|235|237|242|247|248)$", k)])

print("=== AUDIT COLUMNS ===", f249.columns.tolist())
print("=== PAIRED ROLLOUT REQUEST ===")
print(json.dumps({"type": "dose_edit", "model": "both",
                  "split": "tuning", "stays": ids249}))
```