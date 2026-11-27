I'm resuming after a restart. My notes stop at run 212, but the counters say 249 Python runs and 5 forward passes are used. So runs 213–249 and the work behind fwd_4 and fwd_5 are not written down. The last output shows I was checking how the tool applies the dose edit internally: which bin side it uses for rate and dose, and whether z picks up the edit. The state errors are about 1e-3 in every variant, so that check gave no clear answer.

Before spending any more forward passes, I'll rebuild the missing history:
- read the full notebook;
- list what is in fwd_4 and fwd_5, and print any summaries;
- list the kernel variables;
- if the kernel keeps an input history, show the last few runs.

If this shows that fwd_4/fwd_5 were dose_edit runs of the edited model, I can read off the current rollout gains directly and move to calibrating the kappas. If they were states or diagnostic runs, I still need to plan a calibration dose_edit.

```python
import os, json, glob
txt = open('/work/notes.md').read()
print('notes len', len(txt))
print(txt[-1200:])
print('=================== FWD DIRS')
for d in sorted(glob.glob('/work/fwd_*')):
    fl = os.listdir(d)
    print(d, fl)
    for f in fl:
        p = os.path.join(d,f)
        if f.endswith(('.json','.txt','.md')):
            print('  --', f, open(p).read()[:1500])
        elif f.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(p)
            print('  --', f, df.shape, list(df.columns)[:12])
            print(df.head(3).to_string()[:800])
print('=================== GLOBALS')
g = sorted(k for k in globals() if not k.startswith('_'))
print(len(g)); print(g)
if 'In' in globals():
    print('=================== HISTORY len', len(In))
    for i in range(max(1,len(In)-14), len(In)):
        s = In[i]
        print(f'--- In[{i}] len {len(s)}'); print(s[:700])
```