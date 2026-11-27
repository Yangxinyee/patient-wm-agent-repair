The dose signal sits only in low-variance directions of the head input. The top 64 principal components carry almost none of it, so any useful decoder needs large weights. The trade-off is:

| Decoder | Held-out R² | Slope | Weight norm | Shift under 1% noise |
|---|---|---|---|---|
| ridge 0.01 | 0.68 | 0.68 | 105 | 0.82 u/h |
| ridge 0.1 | 0.62 | 0.57 | 45 | 0.35 u/h |
| ridge 1 | 0.47 | 0.38 | 18 | 0.16 u/h |

Random rounding noise is tolerable. It hits both rollout arms alike and averages out over 20 paths and hundreds of stay-hours. The real danger is a systematic bf16 shift, and only the GPU can show that. The factual-arm bias at hour 13 measures it directly.

So I will take ridge 0.1 as the compromise and test it. Norepinephrine is deferred. Its decoders (R² about 0.15–0.2) are too weak for a channel that is unbiased and affordable, and insulin is the main readout.

E3-ins as built:
- **Fit.** Decoder and gates are fitted on all 400 stays. The gates use ridge 1, so their weights stay small.
- **Gate scale.** κ = 6, λ = 4, with every edited value pre-rounded to bf16.
- **Strength.** C = −3.32 per u/h of true dose surprise. That is the summed lag-1 to lag-3 shortfall, carried on the lag-1 channel. The decoder output is rescaled by its cross-validated slope of 0.569.

I will check the in-sample offline NLL and bf16 bias, then save the edit for a clean rollout test.

```python
ws_f, bs_f = zridge(np.ones(len(rd_), bool) & mk, e1, 0.1)
wfp_f = zridge(np.ones(len(rd_), bool), glu.astype(float), 1.0) + zridge(np.ones(len(rd_), bool), pres_i.astype(float), 1.0)
k_f = 0.569
def build_ins3(C, ws_, bs__, k_, wfp, lam=4.0, kap=6.0, units=(31,143), Vref=147.0):
    wf_a, bf_a, wp_a, bp_a = wfp
    W1 = W1o.astype(np.float64).copy(); b1 = b1o.astype(np.float64).copy(); W2 = W2o.astype(np.float64).copy(); b2 = b2o.astype(np.float64).copy()
    ua, ub = units; cs = 100*(C/k_)/(lam*Vref)
    Kw = bfr(kap*(wf_a + wp_a)); Kb = bfr(kap*(bf_a + bp_a - 1.5))
    W1[ub] = Kw; b1[ub] = Kb
    W1[ua] = bfr(Kw + cs*ws_); b1[ua] = bfr(Kb + cs*bs__)
    t_ = bfr(lam*np.r_[cen_gf.mean()/100, cen_gf/100])
    W2[:, ua] = t_; W2[:, ub] = -t_
    return [x_.astype(np.float32) for x_ in (W1, b1, W2, b2)]
E3 = build_ins3(-3.32, ws_f, bs_f, k_f, wfp_f)
nll_e, mean_e = eval_E(*E3); _, mean_e16 = eval_E(*E3, dt=torch.bfloat16)
d_ = nll_e - nll_o
print('E3 in-sample dNLL all %.5f | glu&pres %.4f glu&abs %.4f map %.4f' % (d_.mean(), d_[mk].mean(), d_[glu&~pres_i].mean(), d_[mapf].mean()))
print('bf16-fp32 bias glu&pres %.3f sd %.2f | glu&abs %.3f | map %.3f' % ((mean_e16-mean_e)[mk].mean(), (mean_e16-mean_e)[mk].std(), (mean_e16-mean_e)[glu&~pres_i].mean(), (mean_e16-mean_e)[mapf].mean()))
rr_ = rd_.observed.values - mean_e
cf3, *_ = np.linalg.lstsq(np.c_[e1, e2, e3, np.ones(len(e1))][mk], rr_[mk], rcond=None)
print('resid slopes lag1-3 after E3 (in-sample):', np.round(cf3[:3],2))
for nm_, T_ in zip(['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias'], E3):
    save_edit(nm_, torch.tensor(T_))
print(edited())
with open('/work/notes.md','a') as f:
    f.write("- E3-ins (runs 202-207): units (31,143) pair a=k(G-1.5)+s, b=k(G-1.5), k=6, lam=4, tilt t=cen_gf/100, all values pre-rounded to bf16. G = glucose-family gate + insulin-presence(ins_1>0) gate (ridge a=1, z-aug; clean margins). s from eps_ins_1 decoder ridge a=0.1 (cvR2 .59, slope .569, |w| 45), rescaled 1/k, C=-3.32/u/h (sum of factual deficits lag1-3: -1.6,-1.33,-0.39). NE untouched (decoders R2<.2 too costly).\n")
```