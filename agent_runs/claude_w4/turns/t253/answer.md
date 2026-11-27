The last run (224) crashed inside my helper `u_for`: it compared a value that was `None` (probably a missing rate) with a number. That is a bookkeeping bug, not a finding about the model. My notebook stops at the plan for fwd_6, but 7 forward passes have been used. So fwd_6 and fwd_7 ran, and runs 205–224 were never written up.

Before I do anything else I need to get my bearings back:
- read the fwd_6 and fwd_7 outputs (presumably the first test of the v3 head-1 carrier edit);
- see what the current edit is;
- see which helper functions and arrays are still alive in the kernel.

If fwd_6 shows the v3 edit moving the dose-edit shifts toward the truth (insulin ×2 ≈ −13, NE ×2 ≈ +5 on sel100), I should continue calibrating gains. If not, the carrier approach needs rethinking.

```python
import os, json, glob, types, numpy as np, pandas as pd
nt=open('/work/notes.md').read(); print(len(nt)); print(nt[-600:])
for k in range(1,8):
    d=f'/work/fwd_{k}'
    fs=sorted(os.listdir(d)); print('fwd',k,fs)
    for f in fs:
        p=os.path.join(d,f)
        if f.endswith(('.json','.txt','.md')):
            print('   ',f,':',open(p).read()[:1200].replace('\n',' '))
print('edited',edited())
fn=sorted(k for k,v in list(globals().items()) if isinstance(v,types.FunctionType) and not k.startswith('_'))
print('FUNCS',fn)
arr=sorted((k,tuple(v.shape)) for k,v in list(globals().items()) if hasattr(v,'shape') and not k.startswith('_') and not isinstance(v,type))
print('ARRS',arr[:400])
print('OTHER',sorted(k for k,v in list(globals().items()) if isinstance(v,(dict,list,float,int)) and not k.startswith('_'))[:300])
print(open('/work/sel100_both.json').read()[:300])
```