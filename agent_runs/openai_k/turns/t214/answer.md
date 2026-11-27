The records support immediate effects of about −2 glucose units per insulin-rate unit and +3 arterial-MAP units per norepinephrine-rate unit, with effects persisting across hours. The original rollout is much less responsive, especially to norepinephrine. I’ll audit the reconstructed prediction heads against official predictions; a close match is necessary before trusting another locally calibrated edit.

```python
import re
from pathlib import Path

print("=== VALIDATED LATENT-CHANNEL CANDIDATE HISTORY ===")
sec196 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
for s in sec196:
    if re.match(r"## (?:Run[s]? (?:13[0-9]|14[0-4])\b|Forward)", s):
        if any(w in s.lower() for w in ("139", "factual", "rollout", "posterior", "compress", "reserved")):
            print(s[:700])

print("\n=== CANDIDATE OBJECTS ===")
for k in ("candidate139", "candidate193", "backup193", "zero_candidate", "zero139",
          "candidate180", "best188", "W0", "num_spec0", "sx"):
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "keys", list(v)[:18], "length", len(v))
        if k in ("candidate139", "candidate193", "backup193"):
            for a, b in v.items():
                print(" ", a, type(b).__name__, getattr(b, "shape", None))
    else:
        print(k, type(v).__name__, getattr(v, "shape", None))

print("\n=== RETAINED PREDICTION-HEAD HELPERS ===")
helper196 = []
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and re.search(r"head|predict|view|factual|attr|replay|construct|candidate", k, re.I):
        helper196.append(k)
        c = v.__code__
        print(k, "args", c.co_varnames[:c.co_argcount], "uses", c.co_names[:35])
print("\n=== HEAD/STATE DATA OBJECT NAMES ===")
keys196 = []
for k, v in list(globals().items()):
    if (not callable(v) and not k.startswith("_")
        and re.search(r"head|factual|states|context|prior|predicted|register", k, re.I)):
        if isinstance(v, dict):
            keys196.append((k, "dict", list(v)[:7]))
        elif hasattr(v, "shape"):
            keys196.append((k, type(v).__name__, tuple(v.shape)))
        elif isinstance(v, (list, tuple)):
            keys196.append((k, type(v).__name__, len(v)))
for row in keys196[:65]:
    print(row)

print("\n=== HEAD TENSOR NAMES ===")
for n in NAMES:
    if any(w in n for w in ("z_head", "z_prior", "query_norm", "last_numeric", "type_emb", "attr_heads.value", "embed_transition.value", "last_status")):
        print(n, SHAPES[n])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 196: official-head audit\n"
            "- Recovered record-based immediate and dynamic effects; original rollout is substantially under-responsive.\n"
            "- Candidate193 local diagnostics were refuted by official rollout. Auditing exact head reconstruction before another calibrated edit.\n"
            "- Python runs: 196; forward passes: 12. Saved edits unchanged.\n")
```
