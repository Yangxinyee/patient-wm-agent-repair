The last run was a consistency check. Predicted means computed locally from the stored sel1 states (X) match the tool's predictions for the sel1 rows (MAE 0.74, at the noise floor). They do not match the second batch "A" (MAE 19). That is expected, because batch A has its own state matrix XA (4629 bundles).

My notebook stops at fwd_6, so runs 96–104 and forward passes 7–9 are not written up. Before going on I need to recover what fwd_7–9 contain and what the kernel holds. I will list the forward-pass folders and their files, the kernel's variables and functions, and the functions in /work/lib.py.

If XA comes from one of fwd_7–9 and the batch-A reading rows can be aligned with it, I can go back to the plan in the notebook. That plan is a value-head correction driven by a dose-deviation readout, first checked offline on two sets of stays.

```python
import os, glob, json, types, numpy as np, pandas as pd
for d in sorted(glob.glob('/work/fwd_*'), key=lambda s:int(s.split('_')[-1])):
    fs=sorted(os.listdir(d)); print(d, fs)
    for f in fs:
        p=os.path.join(d,f)
        try:
            if f.endswith('.csv'):
                df=pd.read_csv(p); st=df['stay'].nunique() if 'stay' in df else (df['stay_id'].nunique() if 'stay_id' in df else -1)
                print('   ',f,df.shape,'stays',st,'cols',list(df.columns)[:14])
            elif f.endswith('.npz'):
                z=np.load(p,allow_pickle=True); print('   ',f,{k:z[k].shape for k in z.files})
            elif f.endswith('.json') or f.endswith('.txt') or f.endswith('.md'):
                print('   ',f,open(p).read()[:600].replace('\n',' | '))
        except Exception as e: print('   ',f,'ERR',e)
fn=[];va=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v,types.ModuleType): continue
    if callable(v) and not hasattr(v,'shape') and not isinstance(v,(dict,list)): fn.append(k); continue
    if hasattr(v,'shape'): va.append(f"{k}:{type(v).__name__}{tuple(v.shape)}")
    elif isinstance(v,(dict,list,tuple)): va.append(f"{k}:{type(v).__name__}[{len(v)}]")
    elif isinstance(v,(int,float,str)): va.append(f"{k}={str(v)[:30]}")
print('FUNCS',fn)
print('VARS',va)
src=open('/work/lib.py').read().split('\n'); print('lib.py lines',len(src))
print([l for l in src if l.startswith('def ') or l.startswith('class ')])
```