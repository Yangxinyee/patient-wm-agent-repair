Now the comparison. For drug bundles: measured z-posterior change (edited − original) against my designed `dzD`, via correlation and slope. For all bundles: size of the change in `out` and in each h slot, relative to their natural spread. This separates "units don't fire" (measured dz ≈ 0) from "z is a weak channel" (dz as designed, but out/h barely move).

```python
So=np.load('/work/fwd_10/states_original.npz'); Se=np.load('/work/fwd_10/states_edited.npz')
xo,xe=So['x'],Se['x']; drug=(So['has_insulin']|So['has_norepinephrine']).astype(bool)
print('rows',len(xo),'drug',drug.sum(),'ins',So['has_insulin'].sum(),'ne',So['has_norepinephrine'].sum())
dz=xe[:,960:992]-xo[:,960:992]
print('|dz| drug %.4f nondrug %.4f'%(np.linalg.norm(dz[drug],axis=1).mean(), np.linalg.norm(dz[~drug],axis=1).mean()))
if drug.sum()==len(dzD):
    a=dz[drug].ravel(); b=dzD.ravel(); print('corr vs design %.3f slope %.3f |design| %.4f'%(np.corrcoef(a,b)[0,1], a@b/(b@b), np.linalg.norm(dzD,axis=1).mean()))
for nm,(s0,s1) in [('out',(0,192))]+[('h%d'%s,(192+192*s,384+192*s)) for s in range(4)]+[('z',(960,992))]:
    d=xe[:,s0:s1]-xo[:,s0:s1]; nat=xo[:,s0:s1].std(0).mean()
    print(nm,'mean|d| %.4f  nat sd %.3f'%(np.abs(d).mean(), nat))
```