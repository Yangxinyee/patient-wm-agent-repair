The register layout matters a lot. Putting z_last_value in column 6, age in column 3, and constant 1s in columns 0 and 4 cuts the error to 2.2 mg/dL for glucose and 0.9 mmHg for MAP. That is still short of exact.

Two things could explain what remains:
1. Using the prior mean of z instead of averaging probabilities over random prior draws, as the tool does.
2. A wrong set of always-on flags, or wrong scaling of the age input.

I will test both:
1. Rerun the best layout averaging probabilities over 64 prior draws.
2. Search over which columns are constant 1 for readings, with z_value fixed in column 6 and the age column and age units varied.

If averaging over z draws closes most of the gap, the layout is essentially right; if not, the flag/age search should show it.

```python
def eval_full(lay_const, age_col, zv_col, age_vals, st=0, nsamp=0, seed=0, fx_q=None):
    inp = torch.zeros(len(DT),8,dtype=torch.float64)
    for j in lay_const: inp[:,j] = hl
    inp[:,age_col] = age_vals; inp[:,zv_col] = zv
    base_ = T('embed.code.weight')[cid] + bln + T('embed.bins.value.weight')[lb] + T('embed.bins.rate.weight')[0] + inp @ Wln.T + T('last_status.weight')[st]
    if nsamp==0:
        P = torch.softmax(gelu((q1+base_) @ W0v.T + b0v) @ W2v.T + b2v + tr0, -1)
    else:
        g = torch.Generator().manual_seed(seed); P = 0; ls0 = pr0[:,32:]
        for s_ in range(nsamp):
            zz_ = mu0 + torch.exp(ls0)*torch.randn(mu0.shape, generator=g, dtype=torch.float64)
            q_ = lnorm(ctx0 + zz_ @ T('z_head.weight').T + T('z_head.bias') + T('type_emb.weight')[1], 'query_norm')
            P = P + torch.softmax(gelu((q_+base_) @ W0v.T + b0v) @ W2v.T + b2v + tr0, -1)/nsamp
    m_ = (P[:,1:]*cen_mat).sum(1); e_ = (m_-pm_t).abs()
    return round(float(e_[mgf].mean()),3), round(float(e_[mma].mean()),3), m_
print('best layout, z mean :', eval_full([0,4],3,6,lage)[:2])
print('best layout, z 64 samples:', eval_full([0,4],3,6,lage,nsamp=64)[:2])
ages = {'s':torch.log1p(torch.as_tensor(R_.age.values))/10, 'min':torch.log1p(torch.as_tensor(R_.age.values)/60)/10,
        'h':torch.log1p(torch.as_tensor(R_.age.values)/3600)/10, 's_raw':torch.log1p(torch.as_tensor(R_.age.values))}
out_=[]
for an, av in ages.items():
    for ac in range(8):
        if ac==6: continue
        rest = [j for j in range(8) if j not in (ac,6)]
        for r in range(0,len(rest)+1):
            for S in itertools.combinations(rest, r):
                a_,b_,_ = eval_full(list(S), ac, 6, av)
                out_.append((a_+b_, a_, b_, an, ac, S))
out_.sort(key=lambda x:x[0]); print(out_[:6])
```