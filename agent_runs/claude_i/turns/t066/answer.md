No context formula gets past about 0.7 correlation on both bundle types at once, so the context alone isn't the problem. An error this structural (0.3–0.7) more likely comes from:
- how the 64 posterior outputs are laid out (interleaved mean/log-sd instead of two halves);
- what e_k is for a bundle with no measurements (the empty-set encoding rather than zeros);
- the measurement unit-row mapping.

Next, I will test these cheaply on the existing arrays. The candidates are interleaved output slots, empty-set e_k for drug-only bundles, and the 24 permutations of the four used unit rows for measurement-only bundles. Correlation near 1 for one option identifies it. If nothing works, I'll stop chasing an exact core replica and switch to a strategy that needs only the validated head plus forward passes.

```python
# 1) interleaved layout
zo_def = mlp2(np.c_[ctxb, E0, U0], 'z_post')
for lay_nm, zz_ in [('first32', zo_def[:,:32]), ('even', zo_def[:,0::2]), ('odd', zo_def[:,1::2])]:
    print(lay_nm, [round(np.corrcoef(zz_[kinds==kk].ravel(), tgt[kinds==kk].ravel())[0,1],4) for kk in ['meas','drug']])
# 2) empty-set e for drug-only bundles
pre='encode'
s0 = Tn(pre+'.size.weight')[:,0]*np.log1p(0) + Tn(pre+'.size.bias')
e_empty = lnorm(mlp2(s0[None,:], pre+'.mlp2')[0], Tn(pre+'.norm.weight'), Tn(pre+'.norm.bias'))
E1 = E0.copy(); E1[kinds=='drug'] = e_empty
zo1 = mlp2(np.c_[ctxb, E1, U0], 'z_post')
print("empty-e drug corr", round(np.corrcoef(zo1[kinds=='drug',:32].ravel(), tgt[kinds=='drug'].ravel())[0,1],4))
# 3) unit permutations for measurement bundles (subset for speed)
sub = np.where(kinds=='meas')[0][:600]
best_u=[]
for perm in itertools.permutations([2,3,4,5],2):
    cfg_ = dict(cfg0); cfg_['unit'] = dict(cfg0['unit']); 
    for c_ in ['map_cuff','map_arterial']: cfg_['unit'][c_]=perm[0]
    for c_ in ['glucose_finger','glucose_lab']: cfg_['unit'][c_]=perm[1]
    zo_, tg_, _, _ = zpost_check(cfg_, sub)
    best_u.append((round(np.corrcoef(zo_[:,:32].ravel(), tg_.ravel())[0,1],4), perm))
print(sorted(best_u, reverse=True)[:5])
```