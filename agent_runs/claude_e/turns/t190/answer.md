The notebook stops at the plan to calibrate `gated3(-1.9,3)` with a dose_edit pass. But `fwd_10` and `fwd_11` exist and are not in the notes, so I don't know what they showed. The last run was building a gate for "norepinephrine in the previous bundle" on MAP rows. Its printout is unclear: the gate scores of rows with and without NE seem to overlap, and I can't tell the sign convention without seeing the code.

Before changing anything I will look at the current state:
- the contents of every `fwd_k` folder, and a summary of `fwd_10` and `fwd_11`;
- the kernel's functions and variables, and the input history if the kernel keeps one;
- `/work/edited`.

If `fwd_10` or `fwd_11` is a dose_edit pass with the gated edit, its insulin ×2 shift tells me how far the edit is from my truth estimate (−13.3 to −14.3).

```python
import os, glob, json, sys
import numpy as np, pandas as pd
print('IPython hist:', 'In' in globals(), '_ih' in globals())
fn = sorted([k for k,v in list(globals().items()) if callable(v) and not k.startswith('_') and getattr(v,'__module__',None)=='__main__'])
print('functions:', fn)
vs = sorted([k for k,v in list(globals().items()) if not k.startswith('_') and not callable(v) and type(v).__name__!='module'])
print('vars:', vs)
for d in sorted(glob.glob('/work/fwd_*'), key=lambda s:int(s.split('_')[-1])):
    print(d, sorted(os.listdir(d)))
print('edited dir:', sorted(os.listdir('/work/edited')))
for k in [10, 11]:
    d = f'/work/fwd_{k}'
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.endswith('.json'):
            print(k, f, open(p).read()[:800])
        elif f.endswith('.csv'):
            df = pd.read_csv(p)
            print(k, f, df.shape, list(df.columns))
            print(df.head(2).to_string()[:1200])
            for c in df.columns:
                if df[c].dtype == object and df[c].nunique() < 12:
                    num = [x for x in df.columns if df[x].dtype != object and x not in ('stay','hour')]
                    print('  groupby', c, df.groupby(c)[num].mean().round(3).to_string()[:1500])
print(open('/work/lib.py').read()[:1500])
print('notes chars', len(open('/work/notes.md').read()))
```