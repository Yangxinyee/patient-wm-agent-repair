The slots are essentially identical: median relative difference 1e-4, correlation 1.000. So the recurrent state is one 192-dimensional vector h, replicated four times. It has a large norm (median about 842), and out (norm about 13.9) is just its layer-normalised readout. h behaves like a large accumulator that each bundle's injection adds to.

The factual file has one row per reading (stay, hour, event, observed, predicted mean, NLL), so I can align readings with state rows.

My current hypothesis is that the selective place to act is not the state level but the per-step injection into h. That is `b_in`'s output, whose output space is h's coordinates.

The next step uses the pass-5 pair (dose columns x5 against the original). It takes the immediate injection difference δ_k = Δ_k − Δ_{k−1} at drug bundles, to find the h-direction each drug's dose pushes along (d_ins, d_ne), and compares it with the natural step-to-step change along the same direction at reading bundles. The hypothesis is supported if δ is concentrated in a few directions, d_ins and d_ne are distinct, and the natural per-step change along them at reading bundles is small relative to the dose-induced one. It is refuted if reading-bundle steps vary as much or more along these directions.

```python
h_o = Xo[:,192:384]; h_e = Xe[:,192:384]
Dh = h_e - h_o
first = np.r_[True, sid5[1:]!=sid5[:-1]]
prev_ok = ~first
dstep = np.zeros_like(Dh); dstep[1:] = Dh[1:] - Dh[:-1]; dstep[first] = Dh[first]
nstep = np.zeros_like(h_o); nstep[1:] = h_o[1:] - h_o[:-1]; nstep[first] = 0
io, no_, rd = hi_&~hn_&prev_ok, hn_&~hi_&prev_ok, (~drug)&prev_ok
print('hour sample', np.round(hr5[:8],3))
res_dirs = {}
for lab, msk in [('ins',io),('ne',no_)]:
    U_,S_,Vt_ = np.linalg.svd(dstep[msk], full_matrices=False)
    d = Vt_[0]; res_dirs[lab]=d
    fr = S_[:4]**2/(S_**2).sum()
    sel = [(dstep[m]@d).std() for m in (io,no_,rd)]
    nat = [(nstep[m]@d).std() for m in (io,no_,rd)]
    print(lab, 'top4 frac', np.round(fr,3), '| dstep proj sd ins/ne/read', np.round(sel,2), '| natural step proj sd ins/ne/read', np.round(nat,2))
print('cos(d_ins,d_ne) = %.3f' % abs(res_dirs['ins']@res_dirs['ne']))
print('dstep rms norm ins %.2f ne %.2f read %.2f ; natural step rms norm read %.2f' % tuple(np.sqrt((x**2).sum(1)).mean() for x in (dstep[io], dstep[no_], dstep[rd], nstep[rd])))
```