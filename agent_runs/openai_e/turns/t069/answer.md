The rollout still needs a stronger, better-balanced insulin correction. The previous recovery output was truncated before the relevant helpers appeared, so I’ll retrieve only those objects. This will let me separate intervention sensitivity from factual leakage; independently controllable drug responses would support the proposed low-rank repair.

```python
import inspect
import numpy as np

snapshot61 = list(globals().items())

print("LOCAL HELPER SIGNATURES")
for name61, obj61 in snapshot61:
    if inspect.isfunction(obj61) and obj61.__module__ == "__main__":
        try:
            print(name61, inspect.signature(obj61))
        except Exception:
            print(name61)

print("\nRECENT EDIT OBJECTS")
suffixes61 = ("45", "46", "47", "51", "52", "56", "57", "58")
for name61, obj61 in snapshot61:
    if not name61.endswith(suffixes61):
        continue
    if isinstance(obj61, np.ndarray):
        print(name61, obj61.shape)
    elif isinstance(obj61, dict):
        print(name61, "dict", list(obj61)[:8])
    elif isinstance(obj61, (int, float, str)):
        print(name61, repr(obj61)[:100])

print("\nCAUSAL OBJECT NAMES")
print([name61 for name61, obj61 in snapshot61
       if any(s in name61.lower() for s in
              ["rd_", "dml", "impulse", "beta", "rho", "persist", "ne_", "insulin"])
       and not inspect.ismodule(obj61)])

print("\nSMALL DICTIONARY INVENTORY")
for name61, obj61 in snapshot61:
    if isinstance(obj61, dict) and not name61.startswith("__") and not name61.endswith(suffixes61):
        print(name61, len(obj61), list(obj61)[:5])

print("\nCANDIDATE FILE")
print([(k, candidate_file60[k].shape) for k in candidate_file60.files])
print("\nROLLOUT FILE")
print(rollout60.columns.tolist())
print(rollout60.head(2).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 61: compact helper recovery\n"
            "Previous output truncated before helper inventory. Retrieving only "
            "helper signatures, recent edit arrays and causal-object names. "
            "No weight changes. Python runs: 61; forwards: 7.\n")
```
