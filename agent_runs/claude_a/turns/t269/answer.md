Restoring all three extra zeroings barely changes anything: the plain ablation of dims [4, 2, 67] alone costs about +0.0089 on sel2. My earlier notes put plain ablation of [4, 67, 2, 108] at only +0.0023 and +0.0004 on two separate 300-stay sets. Either sel2 is an unusually sensitive sample, or the current ablation (with these dims set up as memory) differs from what I measured then. The judged cost is on unseen stays, so I need an estimate that isn't dominated by one sample. I'll measure the same ablation on the earlier 300-stay sets (or fresh 300-stay train samples if those sets are gone), each against the original model. If it comes out small there (≤0.002), sel2 is just noisy and the design can proceed with a larger evaluation set. If it's large everywhere, these dims are not cheap and I must look for other dims.

```python
Pb=restore(Psc,('n1','qb','vw'))
cand=[k for k in globals() if k.lower().startswith('ds') and isinstance(globals()[k],dict)]
print('ds dicts', cand[:10])
rng=np.random.default_rng(7); allst=adm['stay_id'].values if 'stay_id' in adm else np.unique(tr.iloc[:,0].values)
sets={}
for k in ['dsE','dsF']:
    if k in globals(): sets[k]=globals()[k]
if len(sets)<2:
    for i_,k in enumerate(['dsX','dsY']): sets[k]=build_ds(tr, list(rng.choice(allst,300,replace=False)), 0)
for k,dsv in sets.items():
    a=tf_eval(PO,dsv,False); b=tf_eval(Pb,dsv,False)
    a0=a[0] if isinstance(a,(tuple,list)) else a; b0_=b[0] if isinstance(b,(tuple,list)) else b
    print(k, 'n=%d base %.4f  d_ablate=%+.5f'%(len(a0), float(a0.mean()), float((b0_-a0).mean())))
```