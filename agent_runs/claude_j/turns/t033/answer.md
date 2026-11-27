The original model gets the direction of all four effects right but the size badly wrong. It predicts insulin x2 lowers glucose_finger by only 2.7 mg/dL and insulin x1/2 raises it by 1.4. For norepinephrine x2 and x1/2 it predicts MAP changes of +0.7 and -0.6 mmHg.

From my impulse responses, about 2 mg/dL per unit/h at first and about 12.5 summed, doubled insulin should lower glucose by tens of mg/dL. Next I'll compute the true changes on the same 100 tuning stays, using their recorded doses and the per-stay-hour 'reached' rows. Before that I'll check which variables survived in the kernel so I don't redo work.

A true insulin x2 shift around -20 mg/dL, far beyond the model's -2.7, would confirm that the insulin effect needs strengthening several-fold. A true change close to the model's would refute that.

```python
import types
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_') and not isinstance(v, types.ModuleType) and not callable(v)}
print(g)
print([k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__'])
print(open('/data/README.md').read()[:3000])
```