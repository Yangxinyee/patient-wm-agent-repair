The architecture confirms that dose information enters through the latent/state update, while predictions also depend strongly on the last recorded reading. The retained record analyses support immediate effects near −2 glucose units per insulin unit and +3 MAP units per norepinephrine unit, with persistence. I’ll inspect the existing head reconstruction next: agreement with official predictions is needed before using it to design a smaller, amplified correction.

```python
import dis, inspect
from pathlib import Path

print("=== PREDICTION HELPER ===")
print("predictions119 globals:", predictions119.__code__.co_names)
try:
    print(inspect.getsource(predictions119))
except (OSError, TypeError):
    dis.dis(predictions119)

print("\n=== HEAD-RECONSTRUCTION NOTES ===")
parts161 = Path("/work/notes.md").read_text().split("\n## ")
for s in parts161:
    if any(s.startswith("Run " + str(k)) or s.startswith("Runs " + str(k))
           for k in (99, 100, 101, 102, 103, 104, 105, 106, 107,
                     108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119)):
        print("\n## " + s[:850])

print("\n=== LOCAL FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        print(name, str(inspect.signature(obj)))

print("\n=== HEAD AND UPDATE OBJECTS ===")
for name, obj in list(globals().items()):
    if name.endswith(("110", "119", "151", "157", "158")) and not inspect.ismodule(obj):
        if isinstance(obj, dict):
            summary = "keys=" + str(list(obj.keys())[:14])
        elif hasattr(obj, "shape"):
            summary = "shape=" + str(obj.shape)
        elif inspect.isfunction(obj):
            summary = "function"
        else:
            summary = type(obj).__name__
        print(name, summary)

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 161: smaller channel plus explicit readout route\n"
            "- Architecture and retained identification suggest testing a small centered-dose state channel with a separately amplified value-head route.\n"
            "- Recovering actual-head reconstruction and local update helpers; no weights changed.\n"
            "- Python runs: 161; forward passes: 10.\n")
```
