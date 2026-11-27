The last step's output was blocked, so I don't know what it showed. No tensors are edited yet. My plan so far is a head-level correction. It uses decoders vI and vN of the dose from the value-head input, centred on the policy, with gains set against the record-based true effects. Those targets are about −14.2 / +7.1 for insulin and +4.8 / −2.4 for norepinephrine on setA. Before going on I need a compact check of what the kernel and the notebook hold: the key numbers and which variables are present. I will print only short excerpts. If the decoders and replica responses are still in memory, I can go straight to building the edit; if not, I must rebuild them cheaply.

```python
txt=open('/work/notes.md').read()
print('notes chars',len(txt))
lines=txt.split('\n')
keys=['truth','fwd_','readout','original model','TRUE','true eff','estimat','x2','dose_edit']
sel=[l for l in lines if any(k in l for k in keys)]
print('n sel',len(sel))
for l in sel[:30]: print('>',l[:200])
g=globals()
for n in ['vI','vN','EDR','Xf','Rf','cf','lbf','DQf','W1','b1','W2','b2','TR','setA','F2','solve_unit','head_mean','run_stay','gen_samples','G3','H100','DEC','RU0']:
    if n in g:
        v=g[n]; s=getattr(v,'shape',None)
        print(n,type(v).__name__,s if s is not None else (len(v) if hasattr(v,'__len__') else ''))
    else: print(n,'MISSING')
```