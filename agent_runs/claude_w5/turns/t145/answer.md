The remaining error is unbiased (mean signed error about 0.02 mg/dL) and spread out (median 1.1 mg/dL for glucose, 0.46 mmHg for MAP). It is larger without a previous value, at ~60-minute gaps and in the first hours. That pattern could still be pure noise from the tool's 4 random z draws.

To decide, I will measure how much a 4-draw average varies locally. I take several independent sets of 4 draws per reading and compare their spread with my observed errors. If the expected 4-draw error is about the size of what I see, the head is correct. Otherwise the leftover (largest for readings with no previous value) points to a remaining detail such as the empty-register inputs.

```python
def head_P(nsamp, seed, lay_const=[0], age_col=3, zv_col=6, av=None):
    av = ages['s'] if av is None else av
    inp = torch.zeros(len(DT),8,dtype=torch.float64)
    for j in lay_const: inp[:,j] = hl
    inp[:,age_col] = av; inp[:,zv_col] = zv
    base_ = T('embed.code.weight')[cid] + bln + T('embed.bins.value.weight')[lb] + T('embed.bins.rate.weight')[0] + inp @ Wln.T + T('last_status.weight')[0]
    g = torch.Generator().manual_seed(seed); P = 0; ls0 = pr0[:,32:]
    for s_ in range(nsamp):
        zz_ = mu0 + torch.exp(ls0)*torch.randn(mu0.shape, generator=g, dtype=torch.float64)
        q_ = lnorm(ctx0 + zz_ @ T('z_head.weight').T + T('z_head.bias') + T('type_emb.weight')[1], 'query_norm')
        P = P + torch.softmax(gelu((q_+base_) @ W0v.T + b0v) @ W2v.T + b2v + tr0, -1)/nsamp
    return P
Ms = torch.stack([(head_P(4, 100+s)[:,1:]*cen_mat).sum(1) for s in range(8)])
sd4 = Ms.std(0)
for c,mk in [('glucose_finger',mgf),('map_arterial',mma)]:
    exp_mae = float((Ms[:, mk]-m64[mk]).abs().mean())
    print(c, 'sd of 4-sample mean: median', round(float(sd4[mk].median()),2), ' expected |4s-64s|', round(exp_mae,2),
          ' observed |tool-64s|', round(float((m64-pm_t)[mk].abs().mean()),2))
```