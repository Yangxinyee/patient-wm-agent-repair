Pass 31 was a factual run on another 100 tuning stays. N5 cost +0.0019 there (1.61959 to 1.62151), mostly from glucose_lab (+0.013) and glucose_finger (+0.005), while map_arterial improved slightly (-0.003). Run 218 must have pooled three 100-stay sets: about +0.0021 ± 0.0008, safely within budget.

N5 stays as the fallback. Its readouts are still far from my record-based estimates of the truth: -2.3 vs -13.6, +1.8 vs +6.8, +0.2 vs +4.5, -0.2 vs -2.2.

Before spending passes I'll recheck which components exist, to see whether any drug-specific tensor is still unexplored. A per-drug embedding or the encode_action 'size' tensor are the candidates. If such a tensor sits after the last normalisation that the drug token passes through, it could amplify drug effects without the saturation that limits input-side scaling. If every drug-specific tensor sits before a LayerNorm, N5 is close to the ceiling.

```python
import re, collections
grp=collections.OrderedDict()
for n in NAMES:
    p='.'.join(n.split('.')[:2]); grp.setdefault(p,[]).append(n)
for p,v in grp.items():
    if any(k in p for k in ['embed','encode_action','prelude']) or len(v)<4:
        print(p, [(x.split('.',2)[-1], tuple(SHAPES[x])) for x in v][:12])
    else:
        print(p, len(v))
```