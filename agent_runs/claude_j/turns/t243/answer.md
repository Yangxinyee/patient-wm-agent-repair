Run 215 shows that `amp_eval` returns per-stay tuples of logit tensors and takes about 0.2 s per stay. I couldn't read the effect of lam from the truncated print, and the result may have come from a cache. The replica is fast, which is good news: I can test edits on many stays cheaply.

Next I will check the structure of the reading tables (`RTs`) and of the `amp_eval` tuples. Then I can write my own clean, saved evaluation for a candidate edit. It will report the factual log loss per reading and the teacher-forced counterfactual glucose and MAP shifts. This step only gathers information. It succeeds if I can identify the columns for event, hour and observed bin, and the logit shapes.

```python
s0=st[0]; rt0=RTs[s0]
print(type(rt0).__name__, getattr(rt0,'shape',None), list(rt0.columns)[:20] if hasattr(rt0,'columns') else '')
r0=r_[0]; print(len(r0),[ (type(x).__name__, tuple(x.shape) if hasattr(x,'shape') else x) for x in r0])
print(type(CEN).__name__, (list(CEN.keys())[:6] if isinstance(CEN,dict) else getattr(CEN,'shape',None)), type(EVV).__name__, str(EVV)[:120])
```