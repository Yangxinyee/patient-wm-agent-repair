The last run looked inside the core that updates the model's state:
- The matrix `c_out` is far from the identity.
- The state slots decay on very different timescales, from about 0.16 h to more than 10,000 h.
- It also listed the tensor names of the prelude and the single core block (self-attention, cross-attention, FFN, norms).

Taken together with my notes, the conclusions so far are:
- The original model's dose responses are about 6–12 times weaker than the effects implied by the records.
- The state hardly records how far a dose departs from what the dosing policy would give in that situation.
- So a change that touches only the readout head will not work. The edit has to put a dose memory into the state update, or into a path that feeds the head.

Next, I will re-read ARCHITECTURE.md in full and list the helper functions still in my kernel. I need the exact routes by which the drug inputs (u), z and the state slots reach the value head before I design that memory.

This step tests no hypothesis. It succeeds if the document shows a route where a dose memory can be added without changing factual predictions, for example the dead z_post units, slow-decay slot dimensions, or a head input.

```python
import re, json, os
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
print(txt[:14000])
fns = sorted([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None) in (None,'__main__')])
print('FUNCS:', fns)
print('FILES:', {f: os.path.getsize('/work/'+f) for f in os.listdir('/work') if os.path.isfile('/work/'+f)})
```