The architecture confirms that drug information reaches predictions through a nonlinear recurrent path; a large token edit may therefore saturate rather than increase the effect. The output was truncated before showing normalization and validation results. I’ll retrieve those compactly. A head correction is worth testing only if its inputs and factual predictions can be reconstructed reliably.

```python
import re, dis
from pathlib import Path

def r175_desc(x):
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        return "dict keys=" + str(list(x)[:10])
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__} len={len(x)}"
    if callable(x) and hasattr(x, "__code__"):
        return "function args=" + str(x.__code__.co_varnames[:x.__code__.co_argcount])
    return str(x)[:100]

print("VOCAB keys:", list(VOCAB))
print("numeric:", str(VOCAB.get("numeric"))[:700])
for key, val in VOCAB.items():
    if any(s in str(key).lower() for s in ("norm", "edge")):
        print(key, r175_desc(val))
        if isinstance(val, dict):
            for k in list(val)[:2]:
                print(" sample:", k, str(val[k])[:250])

print("Normalization helper:")
if "num_spec0" in globals():
    print(r175_desc(num_spec0))
    print(dis.Bytecode(num_spec0).dis()[:1700])

print("Recent retained objects:")
for name, obj in list(globals().items()):
    if re.search(r"(170|171|172|173|166|151)$", name) or name.startswith(("r170_", "r171_", "r172_", "r173_")):
        if not name.startswith("__"):
            print(name, r175_desc(obj))

print("Value-head tensor shapes:")
for name in NAMES:
    if any(s in name for s in ("attr_heads.value", "last_numeric", "query_norm", "z_head", "type_emb")):
        print(name, SHAPES[name])

print("Forward 10 files:")
for p in sorted(Path("/work/fwd_10").iterdir()):
    print(p.name)
    if p.suffix in (".json", ".txt"):
        print(p.read_text()[:1200])
    elif p.suffix == ".csv":
        import pandas as pd
        q = pd.read_csv(p)
        print("shape", q.shape, "columns", q.columns.tolist())
        print(q.head(2).to_string(index=False)[:700])

n = Path("/work/notes.md").read_text()
print("EARLY EVIDENCE:", n[:2300])
for tag in ("Run 139", "Run 171", "Run 172", "Run 173"):
    pos = n.rfind(tag)
    if pos >= 0:
        print(tag, n[pos:pos+800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 175: compact recovery after output truncation\n"
            "- Architecture recovered; large recurrent directions remain rejected for saturation.\n"
            "- Retrieving only normalization, recent head interfaces, and saved-candidate validation.\n"
            "- Python runs: 175; forward passes: 10. No weight changes.\n")
```
