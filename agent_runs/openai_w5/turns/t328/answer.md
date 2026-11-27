The architecture confirms that dose effects must survive recurrent updates and sampled-reading feedback; a one-step coefficient is not enough. The recovery output was truncated before the rollout targets and edit details. I’ll retrieve those concisely so the next trial tests a specific mechanism against both causal estimates and the factual constraint.

```python
import re

n284 = open("/work/notes.md").read()
print("=== SENSOR AND IDENTIFICATION CHECKS ===")
p284 = n284.find("## Sensor checks")
print(n284[p284:p284+3300])

print("=== TARGET / PRIVATE-CHANNEL DETAILS ===")
lines284 = [
    line for line in n284.splitlines()
    if re.search(r"target|expected.*shift|provisional.*effect|coordinates|private.*pair|neuron|readout strength", line, re.I)
]
print("\n".join(lines284)[:2800])

items284 = list(globals().items())
print("=== AVAILABLE HELPER FUNCTIONS ===")
functions284 = [
    name for name, obj in items284
    if callable(obj) and hasattr(obj, "__code__") and not name.startswith("_")
]
print(", ".join(functions284)[:1100])

print("=== RELEVANT OBJECTS ===")
descriptions284 = []
for name, obj in items284:
    if name.startswith("_") or callable(obj):
        continue
    if re.search(r"target|causal|effect|coef|strength|tau|pair|private|tuning.*id|ids.*tun|BASE264|center275", name, re.I):
        if isinstance(obj, dict):
            detail = "keys=" + repr(list(obj)[:12])
        elif isinstance(obj, (str, int, float, np.number)):
            detail = repr(obj)[:150]
        elif isinstance(obj, (list, tuple)):
            detail = "len=" + str(len(obj)) + " " + repr(obj[:5])[:100]
        else:
            detail = "shape=" + str(getattr(obj, "shape", None))
        descriptions284.append(name + ": " + detail)
print("\n".join(descriptions284)[:1800])
```
