The Jacobian answer is clear. At an insulin drug bundle, a unit change in any z dimension moves the next glucose prediction by only 0.01–0.03 mg/dL, and MAP by −0.05 to −0.15. Getting −3 mg/dL would take z changes about 100 times their natural scale. So z, as currently wired through `z_in` into the prelude, is nearly ignored for glucose. This fits the model's weak dose response.

The remaining options are:
- an unused z dimension whose `z_in` column I can set freely, so that a deviation computed by dead z_post units pushes the state strongly;
- or reading an injected change in the state with dead units in the value head.

I'll check:
- each z dimension's variation across bundles and its `z_in` and `z_head` column norms, to spot unused dimensions;
- how many value-head and z_prior hidden units are dead;
- how much `out` moves per unit z.

```python
Zp = zz3[:,:32].numpy()
zsd_ = Zp.std(0); zin_n = np.linalg.norm(T64('z_in.weight').numpy(), axis=0); zh_n = np.linalg.norm(T64('z_head.weight').numpy(), axis=0)
o_ = np.argsort(zsd_)
print('z dims by posterior sd:', [(int(i), round(float(zsd_[i]),3), round(float(zin_n[i]),2), round(float(zh_n[i]),2)) for i in o_[:10]])
print('z sd quantiles', np.quantile(zsd_,[0,.25,.5,.75,1]).round(3))
q_vh = []
mu_, ls_ = zprior2(ctxF)
view_ = ctxF + lin2(mu_,'z_head')
qv = F_.layer_norm(view_ + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
pre_vh = lin2(qv + T64('embed.code.weight')[torch.tensor([CID[c] for c in Fm.event])], 'attr_heads.value_bin.0')
act_vh = gelu_(pre_vh).numpy()
print('value head dead units (sd<0.01):', int((act_vh.std(0)<0.01).sum()), ' z_prior dead:', int((gelu_(lin2(ctxF,'z_prior.0')).numpy().std(0)<0.01).sum()))
Jo = []
for k in samp[:60]:
    s_ = int(S3[k]); ks_ = list(np.where(S3==s_)[0]); i0 = ks_.index(k)
    hp_ = torch.tensor(X3[k-1,192:960].reshape(4,192), dtype=WTd) if i0>0 else torch.zeros(4,192,dtype=WTd)
    op_ = torch.tensor(X3[k-1,:192], dtype=WTd) if i0>0 else torch.zeros(192,dtype=WTd)
    dtk = dts[k]; hf = hp_*torch.exp(-dtk*torch.exp(WT['core.flow.a'])); cx = ctx_from(op_, hf, dtk); z0 = zpost_mean(cx, Et[k], Ut[k])
    idx_m = np.arange(max(ks_[0], k+1-128), k+1); ages = (tmin[k]-tmin[idx_m])*60.0
    h0_, o0_ = core_step(hf, Et[k], Ut[k], z0, Et[idx_m], ages, Vdef)
    d_o = []
    for i in range(32):
        zz_ = z0.clone(); zz_[i] += 1.0
        h1, o1 = core_step(hf, Et[k], Ut[k], zz_, Et[idx_m], ages, Vdef)
        d_o.append(float(torch.norm(o1-o0_)))
    Jo.append(d_o)
Jo = np.array(Jo)
print('|d out| per unit z (mean over bundles):', Jo.mean(0).round(2), ' out natural sd norm', np.linalg.norm(X3[:,:192].std(0)).round(2))
```