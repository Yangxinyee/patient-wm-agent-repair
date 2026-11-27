The 48 ops are rank-1 'in' edits on the prelude attention output, each with alpha 3, along saved vectors v001..v048. The newest kernel variables are `hhat`, `df`, `fu`, `gu`, `recs`, `mdl` and `tr_ids`. I'll check whether the vectors are basis vectors (that would make this head-0 ×4 again) or fitted directions. I'll also peek at those recent variables to reconstruct what I was preparing.

```python
V48 = np.stack([np.load('/work/vectors/v%03d.npy'%i) for i in range(1,49)])
print(V48.shape, 'nnz per vec', np.unique((np.abs(V48)>1e-9).sum(1))[:5], 'argmax', np.abs(V48).argmax(1)[:6], '...', np.abs(V48).argmax(1)[-3:])
print('orthonormal err', np.abs(V48@V48.T-np.eye(48)).max())
for nm_ in ['hhat','df','fu','gu','recs','mdl','tr_ids','l']:
    v=globals().get(nm_); 
    try: print(nm_, type(v).__name__, getattr(v,'shape',None) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v), str(v)[:150].replace('\n',' '))
    except Exception as ex: print(nm_, 'err', ex)
```