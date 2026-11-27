I now have every tensor name and shape for the embedding, encoders, latent, prelude, recurrent blocks, coda and gap/age MLPs. I will write a fresh step function from the notebook's exact description.

First I need a few small lookups:
- how the robust z-score and the fine bins are defined;
- the code and event-kind index maps;
- what an END atom looks like in the bundles;
- the initial-state variables;
- the first bundle times.

These pin down the remaining encoding details.

```python
print(VOCAB['numeric_note'][:300]); print(VOCAB['atom_kinds'], VOCAB['event_kinds'])
it0 = VOCAB['numeric'][0]; print({k: (v if k!='fine_bin_edges' else len(v)) for k,v in it0.items()})
print(code_id); print(code_ek); print(ekind_of)
print('fbin', fbin('glucose_finger','value',162.0), fbin('insulin','rate',1.0))
ends = [a for t,ats_ in bb0 for a in ats_ if a.get('kind')==2][:2]; print('END ex', ends)
print('times', [round(t,3) for t,_ in bb0[:6]])
for nm_ in ['h_init','hinit','oinit','o_init','init','h0']:
    if nm_ in globals(): v_=globals()[nm_]; print(nm_, type(v_).__name__, getattr(v_,'shape',None), (float(v_.abs().sum()) if hasattr(v_,'abs') else ''))
```