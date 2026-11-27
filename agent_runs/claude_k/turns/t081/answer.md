The last run looked for value-head hidden units that are never active, so I could reuse one without changing current predictions. There are none. Over 21,904 readings, the most "off" unit (35) still reaches a pre-activation of −1.99. Several others (53, 125, 24, 180) stay below −1.4 but still vary a little. So reusing any unit costs something, and I need to measure that cost before choosing one.

The gate direction does separate the reading types cleanly:
- Glucose readings project to +1.16…+1.26 (code embeddings 6 and 10 at +1.44).
- MAP readings project to −1.46…−1.57 (codes 5 and 7 at −1.28).
- The other codes (8, 9, 11, 12) sit between −0.33 and +0.21.

So a large gain with a threshold can switch a unit on for glucose only, or MAP only. One problem: inside the glucose group the projection still varies by about 0.1. With a large gain, that variation would leak into the correction. Two units that share the gate, with only one also reading the dose probe, would cancel that leak exactly when their outputs are subtracted. This needs two units per drug.

The conversation was restarted, so first I will re-read the architecture description and list what is in my kernel, then design the edit concretely. If the description confirms the value head is (input q) → Linear → GELU → Linear → 17 bins, the paired-unit plan stands. If the head has extra structure (fine bins, per-code views), I will revise it.

```python
import types, numpy as np
arch = open('/model/ARCHITECTURE.md').read()
print("ARCH length", len(arch))
print(arch[:14000])
print("\n==== kernel variables ====")
for k, v in sorted(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'):
        continue
    if callable(v) and not hasattr(v, 'shape'):
        print("fn ", k)
    else:
        shp = getattr(v, 'shape', None)
        ln = len(v) if hasattr(v, '__len__') and shp is None and not isinstance(v, str) else None
        print("var", k, type(v).__name__, shp if shp is not None else (ln if ln is not None else (v if isinstance(v,(int,float)) else '')))
with open('/work/notes.md', 'a') as f:
    f.write("\n## Restart after run 73\n- run 73: no dead value-head units (most negative max pre -1.99 unit 35; 53 -1.84, 125/24 -1.66, 180 -1.47).\n"
            "- gate direction: glucose readings proj 1.16-1.26 (codes 6,10 emb 1.44), MAP -1.46..-1.57 (codes 5,7 emb -1.28), other codes 8,9,11,12 in [-0.33,0.21].\n"
            "- idea: paired units sharing the gate (one reads the probe) so gate variation cancels; needs cost check of repurposing.\n")
```