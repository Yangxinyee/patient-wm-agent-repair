Forward pass 8 tested the saved head edit (E2) on the 100 tuning stays. It moved all four readouts in the right direction, but not far enough:

| Readout | Original | E2 | Target |
|---|---|---|---|
| insulin x2 | -1.73 | -7.17 | about -14.0 |
| insulin x1/2 | +0.91 | +1.47 | about +7.0 |
| NE x2 | -0.08 | +0.46 | about +4.85 |
| NE x1/2 | -0.48 | -1.92 | about -2.43 |

The response is lopsided. Doubling insulin gained about -5.4, but halving it gained only +0.6, where a linear channel would give about +2.7. For norepinephrine the lopsidedness runs the other way. So the gated channel in the value head is not linear in the decoded dose signal. Likely causes are the GELU flattening negative inputs, or the drug-presence gate weakening when the dose is lowered. E2 also exceeds the factual-fit budget offline (+0.0063 against 0.005), so it cannot be submitted as it stands.

Next I will record this in the notebook and list which helper variables and functions survive in the kernel. After that I will compare the per-hour shifts with per-hour targets. If hour 13 (one hour after the first edited dose) is already about right and the shortfall builds up later, the problem is how the effect carries forward. If hour 13 is already too small or too lopsided, the channel itself must be made linear.

```python
import numpy as np, json
s = json.load(open('/work/fwd_8/summary.json'))
e = {k: v['mean_shift'] for k, v in s['results']['edited']['edits'].items()}
o = {k: v['mean_shift'] for k, v in s['results']['original']['edits'].items()}
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 7 (from run 186)\n- Fwd8 = dose_edit both, sel1, E2: original %s; edited %s. Targets ins_x2 -14.0, ins_half +7.0, ne_x2 +4.85, ne_half -2.43. E2 ~half of ins_x2, ins_half barely moves (asymmetric channel), NE x2 small. E2 also over NLL budget.\n" % (o, e))
skip = ('module','type','builtin_function_or_method')
items, funcs = [], []
for k, v in list(globals().items()):
    if k.startswith('_'): continue
    t = type(v).__name__
    if t in skip: continue
    if t == 'function': funcs.append(k); continue
    sh = getattr(v, 'shape', None)
    if sh is None and hasattr(v, '__len__') and t in ('list','dict','tuple'): sh = (len(v),)
    items.append(f"{k}:{t}{'' if sh is None else list(sh)}")
print(e)
print(len(items)); print(' '.join(items)[:5000])
print('FUNCS', ' '.join(funcs)[:2000])
```