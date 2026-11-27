The bin centres are stored per measurement code (map_arterial, map_cuff, glucose_lab, glucose_finger), and class 0 adds nothing to the predicted mean. The drug signals barely move the code-gate directions: the gate shift is about 0.002 per unit rate on a unit-norm direction. Each rate read picks up its own drug with gain 1.00 ± 0.04 and the other drug at about 0. So the pair-unit design can be used.

What I still need is a way to compute the head input q for map rows as well as gf rows. That will give the baselines and gate thresholds.

Next, I will inspect `wc` and `wc_row` (arguments and constants), list the newest arrays in the kernel, and print the code ids and bin centres. If `wc` takes a code argument, I can build q for all four measurement codes directly. If it does not, I will rebuild q from the embedding terms.

```python
import numpy as np, pandas as pd
items = list(globals().items())
tail = [(k, getattr(v,'shape',None)) for k, v in items[-90:] if isinstance(v,(np.ndarray,pd.DataFrame))]
print(str(tail)[:1500])
for fn in ['wc','wc_row']:
    f = globals().get(fn)
    if f is None: print(fn,'missing'); continue
    c = getattr(f,'__code__',None)
    print(fn, type(f).__name__, (c.co_varnames[:c.co_argcount], [x for x in c.co_consts if not hasattr(x,'co_code')][:20], c.co_names[:40]) if c else '')
print('HD', type(HD).__name__, (list(HD.keys())[:30] if isinstance(HD,dict) else ''))
print('ctx', type(ctx).__name__, (ctx.shape if hasattr(ctx,'shape') else str(ctx)[:200]))
print([k for k,v in items if isinstance(v,dict)][-40:])
print([(c['id'],c['name']) for c in VOCAB['codes'] if c['name'] in VC])
print({k: np.round(v,1).tolist() for k,v in VC.items()})
with open('/work/notes.md','a') as f:
    f.write("- Run 278: VC centres dict per code (16 each); mean=sum_{b>=1} p(b)c[b-1]. GD.DQ ~.002/unit (negligible); RD own gain 1.00 sd .04, cross 0; |RD| ins 141, ne 66.\n")
```