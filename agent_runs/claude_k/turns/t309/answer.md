Doubling the input embedding of dose and rate by 6 did not strengthen the model's dose response. The insulin x2 shift fell from -2.15 to -1.70 and the norepinephrine x2 shift from +0.34 to +0.10. So the bottleneck is downstream, and the input-amplification approach is refuted.

I will revert it so the model is back to the original. With only 15 Python runs left, I must recover two things I lost in the kernel restart:
1. The true-effect estimates I derived earlier from the records. These are the targets.
2. The exact tensor inventory, to find a pathway where a dose-dependent term can be written.

This run therefore:
- drops all edits;
- logs the result;
- prints the head of my notebook (the targets should be there) and keyword lines from its middle;
- lists the kernel helpers and arrays that survive;
- prints all tensor names with shapes, plus the architecture lines on the value head, transition and view.

If the notes contain explicit true-effect numbers, I set those as targets. If not, I will have to re-estimate them from the records with an innovation regression, which costs runs.

```python
import json, os, numpy as np
for n in ['embed.bins.dose.weight','embed.bins.rate.weight','embed.numeric.weight']:
    drop_edit(n)
print('edited now:', edited())
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 7 (run 286)\n- fwd13 dose_edit both tuning100, input-amp k=6: ins x2 -1.70 (orig -2.15), half 1.43 (orig 1.23); ne x2 .10 (orig .34), half -.105 (orig -.057) -> input amp USELESS, REVERTED (no edits now).\n")
txt = open('/work/notes.md').read()
print('notes len', len(txt))
print(txt[:6500])
print('=====KEY LINES (middle)=====')
mid = txt[6500:max(6500,len(txt)-8000)]
keys = ['true','TRUE','ffect','arget','ruth','causal','elieve','stimat','x2','kappa','nstrument','andom','olicy','confound','IRF','impulse']
for l in mid.split('\n'):
    if any(k in l for k in keys): print(l[:240])
g = {k:v for k,v in list(globals().items()) if not k.startswith('_')}
fn = [k for k,v in g.items() if callable(v) and getattr(v,'__module__',None)=='__main__']
print('FUNCS:', fn)
arr = [(k, tuple(v.shape)) for k,v in g.items() if isinstance(v, np.ndarray) and v.size>1]
print('ARRAYS:', arr[:60])
print('NAMES:', ' | '.join(f"{n}{tuple(SHAPES[n])}" for n in NAMES))
arch = open('/model/ARCHITECTURE.md').read()
print('ARCH len', len(arch))
for l in arch.split('\n'):
    if any(k in l.lower() for k in ['value head','transition','view','value_bin','logit']): print('A:', l[:240])
```