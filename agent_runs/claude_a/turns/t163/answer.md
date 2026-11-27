There is no input history, so the code from runs 143–153 is gone; only the compiled functions remain. From now on I will write every edit-building function into /work/edit_lib.py and exec it, so the code survives restarts and can be read.

Before re-implementing the memory edit I need three things:
- which kernel dict holds the original weights;
- the tensor-name layout, compressed by pattern;
- the baseline value of my offline factual-NLL function `tf_nll`, and how long it takes, since it will be my main check on factual fit.

I expect `PO` to equal the original weights, and `tf_nll(PO)` to come out near the tool's 1.4039 on sel2 if my replica is exact.

```python
import re, time
ok={}
for nm in ['P','PO','PS','P_','Po','W','Wt','D_','TJ']:
    v=globals().get(nm)
    if isinstance(v,dict) and 'core.c_out.weight' in v:
        d=0.0
        for n in NAMES:
            a=v[n]; a=a if torch.is_tensor(a) else torch.tensor(np.asarray(a))
            d=max(d,float((a.float()-load_original(n)).abs().max()))
        ok[nm]=d
print('max diff vs original:',{k:round(v,5) for k,v in ok.items()})
pat={}
for n in NAMES:
    k=re.sub(r'\.\d+\.','.#.',n); pat.setdefault(k,[]).append(tuple(SHAPES[n]))
print(len(pat),'patterns')
print(' ; '.join(f"{k}{v[0]}x{len(v)}" if len(v)>1 else f"{k}{v[0]}" for k,v in pat.items()))
t0=time.time(); r=tf_nll(PO); print('tf_nll(PO)', r if np.isscalar(r) else type(r), '%.1fs'%(time.time()-t0))
```