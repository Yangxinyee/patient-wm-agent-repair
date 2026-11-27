**What the last result showed:** Run 291 saved one edit, which is still the only edit. It multiplies the two dose/rate columns of the numeric input embedding (`embed.numeric.weight`) by 3. This amplifies the model's own learned dose pathway. The tool has not tested it yet.

**What I conclude:** The head-level edits all broke on the tool, and I have dropped them. So this edit is the one live candidate. Before testing it I need three things:
- the setA stay ids, because forward requests need explicit ids;
- the original model's dose-edit readouts on setA, from my earlier passes, to set against my record-based targets (−14.17, +7.09, +4.82, −2.41);
- what the numeric features are, and whether insulin and norepinephrine share these columns.

I have only 9 Python runs left. So this one run gathers all of that at once: the notebook sections I can't see, the setA ids, summaries of every earlier forward pass, and the current edit's column ratios.

**What would confirm or refute my plan:**
- *Confirm:* the original model's insulin x2 readout has the right sign but is too small. Then scaling the model's own pathway is a sensible lever, and I can calibrate the factor with "dose_edit both" and check cost with "factual both".
- *Refute:* the readout has the wrong sign or is near zero. Then scaling cannot reach the target, and I must rethink.

```python
import json, os, glob
import numpy as np, pandas as pd
txt=open('/work/notes.md').read()
print('notes chars',len(txt))
early=txt[:max(0,len(txt)-8000)]
print(early[:7000])
print('=====SETA')
try:
    sa=json.load(open('/work/setA.json'))
    s=json.dumps(sa); print(type(sa).__name__, len(sa), s[:1600])
except Exception as e: print('setA err',e)
print('=====FWD')
def summ(p):
    df=pd.read_csv(p)
    print('    cols',list(df.columns)[:16],'n',len(df))
    flt=[c for c in df.columns if any(k in c.lower() for k in ('ran','reach','after'))]
    if flt:
        try:
            df=df[df[flt[0]].astype(bool)]; print('    filtered on',flt[0],'n',len(df))
        except Exception as e: print('    filt err',e)
    num=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ('stay','stay_id','hour','h','path')]
    obj=[c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique()<=12]
    if obj:
        for c in obj:
            print('    by',c); print(df.groupby(c)[num].mean().round(3).to_string()[:1400])
    else:
        print(df[num].mean().round(3).to_string()[:900])
for d in sorted(glob.glob('/work/fwd_*'), key=lambda p:int(p.rsplit('_',1)[-1])):
    fs=sorted(os.listdir(d)); print(d,fs)
    for f in fs:
        p=os.path.join(d,f)
        try:
            if f.endswith('.csv'): print('  ',f); summ(p)
            elif f.endswith(('.json','.txt','.md')): print('  ',f, open(p).read()[:600])
        except Exception as e: print('  err',f,e)
print('=====EDIT')
W=np.asarray(load('embed.numeric.weight')); W0=np.asarray(load_original('embed.numeric.weight'))
print('col norm ratio',(np.linalg.norm(W,axis=0)/np.linalg.norm(W0,axis=0)).round(3), 'edited',edited())
arch=open('/model/ARCHITECTURE.md').read().splitlines()
hits=[(i,l) for i,l in enumerate(arch) if any(k in l.lower() for k in ('numeric','z_dose','z_rate','rate bin','bins.rate'))]
for i,l in hits[:30]: print(i,l[:260])
print(list(VOCAB.keys()))
for k in VOCAB:
    if 'norm' in k.lower(): print(k, json.dumps(VOCAB[k])[:900])
with open('/work/notes.md','a') as f:
    f.write('\n## Session 10 (runs 292-)\n- Run 292: re-orientation dump (setA ids, fwd summaries). Current edit: embed.numeric dose/rate cols x3 (untested on tool).\n')
```