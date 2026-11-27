The stronger norepinephrine setting improves immediate dose-response slopes, but that does not establish better rollouts, and its factual-loss estimate is close to the limit. I’ll recover the records-derived effect estimates and audit stay IDs without changing weights. The next GPU tests must show both preserved factual fit and improved rollout effects.

```python
import os, re, json, hashlib
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

notes276 = open("/work/notes.md").read()
print("=== EARLY NOTEBOOK: RECORD-BASED IDENTIFICATION ===")
print(notes276[:18000])
print("=== REMAINING NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes276[18000:].splitlines()
                if line.startswith("#")))

print("=== LIVE FITTING OBJECTS ===")
for name276 in sorted(list(globals())):
    if any(tag in name276 for tag in ("179", "214", "234", "263", "269", "273", "275")):
        obj276 = globals()[name276]
        if isinstance(obj276, np.ndarray):
            desc276 = f"array {obj276.shape} {obj276.dtype}"
        elif isinstance(obj276, dict):
            desc276 = "dict keys=" + repr(list(obj276.keys())[:24])
        elif callable(obj276):
            desc276 = "callable"
            if hasattr(obj276, "__code__"):
                desc276 += " args=" + repr(obj276.__code__.co_varnames[:obj276.__code__.co_argcount])
        else:
            desc276 = type(obj276).__name__
        print(name276, desc276)

print("=== RECENT FORWARD FILES AND STAY IDS ===")
audit_ids276 = {}
for folder276 in ("/work/fwd_24", "/work/fwd_25"):
    print(folder276, os.listdir(folder276))
    for filename276 in sorted(os.listdir(folder276)):
        if filename276.endswith(".csv"):
            frame276 = pd.read_csv(os.path.join(folder276, filename276))
            print(filename276, frame276.shape, list(frame276.columns))
            print(frame276.head(2).to_string(index=False))
            idcol276 = next((c for c in ("stay_id", "stay", "sid", "id")
                            if c in frame276.columns), None)
            if idcol276:
                ids276 = frame276[idcol276].drop_duplicates().tolist()
                audit_ids276[(folder276, filename276)] = ids276
                print("STAY IDS", json.dumps(ids276))

print("=== CURRENT EDIT HASH ===")
hash276 = hashlib.sha256()
for name276 in sorted(edited()):
    value276 = load(name276)
    array276 = value276.detach().cpu().numpy() if hasattr(value276, "detach") else np.asarray(value276)
    hash276.update(name276.encode())
    hash276.update(array276.tobytes())
print(edited(), hash276.hexdigest()[:16])

with open("/work/notes.md", "a") as f276:
    f276.write("\n## Run 276 / restart evidence recovery\n"
               "Read architecture, record schema, early causal-identification notes, "
               "live fitting-object summaries, and recent audit stay IDs. "
               "No weights changed; balanced broadened .25/.25 remains staged "
               "pending GPU factual and paired-rollout validation.\n")
```