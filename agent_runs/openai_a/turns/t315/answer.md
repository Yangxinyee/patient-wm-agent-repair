The saved blend helpers allow a cheaper test: replace part of the gated channel rather than add extra strength. I’ll screen mixtures whose coefficients sum to one and recover the causal estimates. Lower factual loss with a stronger halving response would support testing a mixture on the GPU; higher loss would rule it out.

```python
import inspect
import numpy as np
from pathlib import Path

for name in ["make_blend268", "blend_eval268", "blend_parts268",
             "make_selective_contrast_graft", "make_ne_data",
             "ne_threshold_rd"]:
    f = globals()[name]
    print(name, inspect.signature(f), "names:", f.__code__.co_names)

print("\nBASE BLEND RETURN:")
br271 = blend_eval268(blend_factparts268["first"], 1.0, 0.0)
if isinstance(br271, dict):
    print({k: (np.shape(v), float(np.mean(v))) for k, v in br271.items()
           if isinstance(v, (np.ndarray, float, int, np.number))})
elif isinstance(br271, tuple):
    print([(np.shape(v), float(np.mean(v))) for v in br271])
else:
    print(type(br271), np.shape(br271), str(br271)[:300])

print("\nREPLACEMENT BLEND RETURNS:")
for a271, b271 in [(0.95, 0.05), (0.90, 0.10), (0.85, 0.15),
                   (0.80, 0.20), (0.95, 0.0)]:
    r271 = blend_eval268(blend_factparts268["tuning"], a271, b271)
    if isinstance(r271, dict):
        summary271 = {k: (np.shape(v), float(np.mean(v))) for k, v in r271.items()
                      if isinstance(v, (np.ndarray, float, int, np.number))}
    elif isinstance(r271, tuple):
        summary271 = [(np.shape(v), float(np.mean(v))) for v in r271]
    else:
        summary271 = (type(r271).__name__, np.shape(r271), str(r271)[:150])
    print((a271, b271), summary271)

text271 = Path("/work/notes.md").read_text()
sections271 = text271.split("\n## ")
print("\nCAUSAL ESTIMATE SECTIONS:")
for s271 in sections271:
    title271 = s271.splitlines()[0]
    if any(term in title271 for term in
           ["threshold identification", "Sustained insulin",
            "NE flexible", "Dynamic identification", "NE policy-boundary",
            "NE boundary uncertainty"]):
        print("## " + s271[:1500])
```
