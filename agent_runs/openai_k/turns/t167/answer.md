The local update now matches the official states to numerical precision, and the two dose channels are largely drug-specific. The remaining problem appears to be how those channels affect future readings. I’ll recover the retained interfaces and identification estimates before fitting new directions; improved paired rollouts without exceeding the factual-loss limit would support this approach.

```python
import os, re, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

print("\n=== RELEVANT KERNEL OBJECTS ===")
pat152 = re.compile(
    r"(151|150|149|148|145|144|90|impuls|deconv|candidate|channel|"
    r"local|update|readout|surrog|context|feature|recipe|physical|capture)",
    re.I
)
for name152 in sorted(list(globals())):
    if name152.startswith("_") or not pat152.search(name152):
        continue
    obj152 = globals()[name152]
    if callable(obj152):
        if hasattr(obj152, "__code__"):
            c152 = obj152.__code__
            print(name152, "FUNCTION",
                  inspect.signature(obj152),
                  "uses", c152.co_names)
    elif isinstance(obj152, dict):
        print(name152, "DICT", len(obj152),
              "keys", list(obj152)[:16])
    elif isinstance(obj152, (np.ndarray, torch.Tensor)):
        print(name152, type(obj152).__name__, tuple(obj152.shape))
    elif isinstance(obj152, (list, tuple)):
        print(name152, type(obj152).__name__, "len", len(obj152),
              "first", repr(obj152[:2])[:160])
    elif isinstance(obj152, (str, int, float)):
        print(name152, repr(obj152)[:180])

print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND EDIT DESIGN ===")
notes152 = open("/work/notes.md").read()
sections152 = re.split(r"(?=^## )", notes152, flags=re.M)
chosen152 = [
    s for s in sections152
    if any(x in s.lower() for x in [
        "deconvol", "bootstrap", "identification", "confound",
        "candidates90", "compression", "reserved", "paired-response"
    ])
]
print("\n".join(chosen152[:14])[:16000])

print("\n=== CURRENT EDIT SIZES ===")
for name152 in edited():
    a152 = load(name152)
    b152 = load_original(name152)
    print(name152, tuple(a152.shape),
          "change_norm", float(np.linalg.norm(np.asarray(a152)-np.asarray(b152))))

with open("/work/notes.md", "a") as f152:
    f152.write(
        "\n## Run 152: recover complete routing and identification interfaces\n"
        "- Local original-state reconstruction is numerically exact; retained dose channels are approximately physical and drug-specific.\n"
        "- Inspecting architecture, schema, existing helpers and early identification evidence before a closed-form routing edit.\n"
        "- No weights changed. Python runs: 152; forward passes: 10.\n"
    )
```