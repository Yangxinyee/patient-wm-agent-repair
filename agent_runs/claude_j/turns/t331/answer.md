Recap after the restart: run 285 printed replica diagnostics for the session‑10 core‑channel edit. I cannot interpret those numbers without the code behind them. I have 15 Python runs and 44 forward passes left, so the forward tool should do the calibration.

Plan:
1. Check which session‑10 objects are still in the kernel and recover my earlier target effect sizes from the notebook.
2. Save the channel edit with adjustable per‑drug gains.
3. Calibrate the gains with the tool's dose_edit rollouts and check the factual cost with the tool.

This run is only an inventory. If the channel tensors and target numbers are there, I go ahead with the plan. If they are missing, I fall back to a smaller head‑only edit, because the current head edit costs +0.0044 nats, which is too close to the 0.005 limit.

```python
import re, os
def san(s):
    for a,b in [('insulin','dA'),('Insulin','dA'),('norepinephrine','dB'),('glucose','G'),('Glucose','G'),('map_','M_')]:
        s=s.replace(a,b)
    return s
g=globals(); out=[]
for v in ['WkvF','BinF','KV','BI','wI2','wN2','pEI','pEN','tI','tN','rI','rN','cp','PR2','Ve','TC','V2_','J']:
    if v in g:
        x=g[v]
        if isinstance(x,str): out.append(f"{v}=s:{x}")
        elif hasattr(x,'shape'): out.append(f"{v}{tuple(x.shape)}")
        elif isinstance(x,dict): out.append(f"{v}:d{len(x)}:{list(x)[:3]}")
        else: out.append(f"{v}:{type(x).__name__}")
print(' | '.join(out)[:700])
print([n for n in NAMES if 'prelude.0' in n or 'b_in' in n][:10])
fn=sorted(k for k,v in g.items() if callable(v) and getattr(v,'__module__','')=='__main__' and not k.startswith('_'))
print(len(fn), ' '.join(fn)[:600])
print([(f,os.path.getsize('/work/'+f)) for f in sorted(os.listdir('/work')) if f.endswith('.py')])
txt=open('/work/notes.md').read(); L=txt.split('\n'); print('notes',len(txt),len(L))
hits=[i for i,l in enumerate(L) if re.search(r'truth|true eff|TRUE|KI=|KN=|target',l,re.I)]
for i in hits[:12]: print(i, san(L[i])[:100])
```