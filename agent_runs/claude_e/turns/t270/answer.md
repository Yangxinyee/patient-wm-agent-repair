Fwd_26 was a factual check of the saved edit (ins_only(-2.7), hash 2bc751d207). The value NLL went from 1.62044 to 1.62503, a cost of +0.00459 per reading. Glucose_finger rows got worse (1.446 → 1.4596). Glucose_lab rose slightly (+.0065). MAP rows were unchanged, as expected, since the NE part is not edited.

That cost is about what I predicted for batch tu0 at -2.7, interpolating .00428 at -2.6 and .00492 at -2.8. So I think this batch is tu0, the most expensive batch seen so far. If so, the result agrees with my bootstrap: mean cost about .0029 over 800 stays and a low risk of exceeding .005 on a large set. If it turns out to be a cheaper batch, -2.7 is riskier than I believed and I should step back to -2.6.

Next I will confirm which stays fwd_26 used and check the kernel state (edit hash, helper functions in lib.py). I will also re-read the architecture summary to judge one last time whether any NE fix is feasible. NE is still by far the largest error: x2 is -0.23 against a true +4.5, half is -0.56 against -2.26.

```python
import json, os, hashlib, numpy as np, pandas as pd
fo = pd.read_csv('/work/fwd_26/factual_original.csv'); fe = pd.read_csv('/work/fwd_26/factual_edited.csv')
print(fo.columns.tolist()[:20], len(fo))
sid = sorted(fo['stay_id'].unique()) if 'stay_id' in fo.columns else None
print('stays', (sid[0], sid[-1], len(sid)) if sid else None)
for f in ['sel_f3.json','sel_tu0.json','sel_tu1.json','sel_tu2.json']:
    try:
        s = json.load(open('/work/'+f)); s = s if isinstance(s,list) else s.get('stays',s)
        print(f, len(s), s[:2], 'overlap', len(set(s)&set(sid)) if sid else '')
    except Exception as e: print(f, e)
print('edited:', edited())
h = hashlib.md5(b''.join(load(n).numpy().tobytes() if hasattr(load(n),'numpy') else np.asarray(load(n)).tobytes() for n in sorted(edited()))).hexdigest()[:10]
print('my md5 of edits', h)
print([k for k,v in globals().items() if callable(v) and not k.startswith('_')][:80])
src = open('/work/lib.py').read(); print('lib.py chars', len(src))
print([l for l in src.splitlines() if l.startswith('def ')][:40])
arch = open('/model/ARCHITECTURE.md').read(); print('ARCH chars', len(arch)); print(arch[:5000])
```