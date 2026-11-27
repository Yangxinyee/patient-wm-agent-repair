The best context variant (no time flow, with gap) still leaves a 0.45 error, so the context isn't the main problem. Action-only bundles may use e = encode(empty set), a constant, rather than zero (my old code had an 'empty_const' option). I'll test that on action-only bundles alone. A small error there would show the context is right and point the blame at the measurement encoder; a large one means the context is still wrong.

```python
pre = 'encode'
e_empty = LNd(MLP(LIN(torch.tensor([np.log1p(0.)]), pre+'.size'), pre+'.mlp2'), pre+'.norm')
def zp_act(flow=True, eempty=True):
    errs = []
    for sid in np.unique(F4['stay_id']):
        m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(int(sid))
        for k in range(1, len(B)):
            if not set(B[k].code.astype(int)) <= {8, 9}: continue
            e, u = enc2(B[k])
            if eempty: e = e_empty
            dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)
            if flow: hp = hp*torch.exp(-dt*torch.exp(fa_))
            ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
            zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
            errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return len(errs), round(float(np.median(errs)), 3)
print([(f, ee, zp_act(f, ee)) for f in (True, False) for ee in (True, False)])
```