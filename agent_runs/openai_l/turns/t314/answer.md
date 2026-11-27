The early notes contain a sign-reversed state-space estimate that was explicitly rejected, alongside stronger threshold-based evidence for insulin. I won’t treat that rejected estimate as truth. I’ll retrieve the later identification conclusions and exact audit IDs, then test whether the staged edit improves rollouts without exceeding the factual-loss limit.

```python
print("=== IDENTIFICATION NOTES ===")
print(notes276[4400:10400])

print("=== LATER RECORD-BASED CONCLUSIONS ===")
sections277 = re.split(r"(?m)(?=^## )", notes276)
selected277 = [
    s for s in sections277
    if re.search(r"RD.*target|identified|identification|causal estimates|"
                 r"record.supported.*target|persistence estimate|"
                 r"accepted.*effect|rollout targets|threshold.*estimate", s, re.I)
]
for section277 in selected277[-6:]:
    print(section277[:450])

print("=== AUDIT IDS ===")
for key277, ids277 in audit_ids276.items():
    if key277[0] == "/work/fwd_24" and "edited" in key277[1]:
        print(key277, json.dumps(ids277))
        break

print("=== CURRENT HASH AND AVAILABLE FIT OBJECTS ===")
print(hash276.hexdigest()[:16])
for name277 in sorted(list(globals())):
    if any(tag in name277 for tag in ("263", "269", "273")):
        obj277 = globals()[name277]
        if isinstance(obj277, dict):
            print(name277, "keys", list(obj277)[:16])
        elif isinstance(obj277, np.ndarray):
            print(name277, obj277.shape)
        elif callable(obj277):
            print(name277, "function")

with open("/work/notes.md", "a") as f277:
    f277.write("\n## Run 277 / concise recovery after output truncation\n"
               "Retrieved later identification notes and exact paired-audit IDs. "
               "Rejected early sign-reversed SSM estimates are not treated as truth. "
               "No weights changed.\n")
```