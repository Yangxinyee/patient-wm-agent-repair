The gates separate cleanly on held-out stays. The combined gate value G is:

| Rows | G (1st–99th percentile) |
|---|---|
| glucose, insulin running | 1.94–2.07 |
| glucose, no insulin | 0.92–1.17 |
| MAP, insulin running | 0.93–1.07 |
| MAP, no insulin | −0.09–0.17 |

The presence gate hardly depends on the insulin rate (−0.005 per u/h), so halving a dose won't close it.

The lag-1 surprise decoder generalises: held-out R² is 0.68 and slope 0.675. Its response to the insulin rate at fixed glucose-implied dose is 0.667, which is the quantity the counterfactual needs.

I will use κ = 10. An open gate then sits at about +4.4 to +5.7. A glucose row without insulin sits at about −3.3 or below even at the 99th percentile, so leakage is under 1%.

Now I will build E3-ins and score it offline on the held-out selS stays, using 4 z draws with common random numbers, at two strengths:
- **C = −2.37.** This corrects only the lag-1 shortfall, without bias.
- **C = −4.92.** This carries the summed lag-1 to lag-3 shortfall on the lag-1 channel.

For each I will report:
- the change in NLL overall and by event;
- the bias on rows that should be untouched;
- the bf16 bias, which must be near 0;
- the residual slopes on the insulin surprise afterwards.

C = −2.37 should improve or keep NLL on glucose rows. C = −4.92 should cost about 0.002–0.003 overall. Anything well above that, or a bf16 bias above 1 mg/dL, refutes the design.

```python
cen_gf = np.array(VOCAB['value_bin_centres']['glucose_finger'], dtype=np.float64)
XA4 = aug_wc(rows_all, 4, 7)
def build_ins(C, lam=4.0, kap=10.0, units=(31,143), Vref=147.0):
    W1 = W1o.astype(np.float64).copy(); b1 = b1o.astype(np.float64).copy(); W2 = W2o.astype(np.float64).copy(); b2 = b2o.astype(np.float64).copy()
    ua, ub = units; cs = 100*C/(lam*Vref)
    W1[ua] = kap*(wf + wp) + cs*ws; b1[ua] = kap*(bf_ + bp_ - 1.5) + cs*bs_
    W1[ub] = kap*(wf + wp);         b1[ub] = kap*(bf_ + bp_ - 1.5)
    t_ = np.r_[cen_gf.mean()/100, cen_gf/100]
    W2[:, ua] = lam*t_; W2[:, ub] = -lam*t_
    return [x_.astype(np.float32) for x_ in (W1, b1, W2, b2)]
def eval_E(W1, b1, W2, b2, dt=torch.float32, XAx=None):
    XAx = XA4 if XAx is None else XAx
    pacc = 0
    for d_ in range(XAx.shape[0]):
        x = torch.tensor(XAx[d_]).to(dt)
        hid = torch.nn.functional.gelu(x @ torch.tensor(W1).to(dt).T + torch.tensor(b1).to(dt))
        lg = (hid @ torch.tensor(W2).to(dt).T + torch.tensor(b2).to(dt)).float() + torch.tensor(HI['trans']).float()
        pacc = pacc + torch.softmax(lg, -1).numpy().astype(np.float64)
    p_ = pacc / XAx.shape[0]
    nll = -np.log(np.clip(p_[np.arange(len(ob_)), ob_], 1e-12, None))
    mean = (p_[:,1:]*HI['cen']).sum(1) / p_[:,1:].sum(1)
    return nll, mean
nll_o, mean_o = eval_E(W1o, b1o, W2o, b2o)
_, mean_o16 = eval_E(W1o, b1o, W2o, b2o, torch.bfloat16)
for C_ in [-2.37, -4.92]:
    E_ = build_ins(C_)
    nll_e, mean_e = eval_E(*E_); _, mean_e16 = eval_E(*E_, dt=torch.bfloat16)
    d_ = nll_e - nll_o; b_ = mean_e - mean_o
    print('C=%.2f test dNLL %.5f (train %.5f)' % (C_, d_[tem].mean(), d_[trm].mean()),
          ' by ev:', {e_[:5]: round(d_[tem & (evv==e_)].mean(), 4) for e_ in ['glucose_finger','glucose_lab','map_arterial','map_cuff']})
    print('   bias glu&abs %.3f map %.3f | bf16 bias vs fp32 (glu&pres) %.3f, orig bf16 %.3f' % (
        b_[tem & glu & ~pres_i].mean(), b_[tem & mapf].mean(), (mean_e16-mean_e)[tem & mk].mean(), (mean_o16-mean_o)[tem & mk].mean()))
    rr_ = rd_.observed.values - mean_e; mm_ = tem & mk
    cf2, *_ = np.linalg.lstsq(np.c_[e1, e2, e3, np.ones(len(e1))][mm_], rr_[mm_], rcond=None)
    print('   resid slopes after edit (test present glu):', np.round(cf2[:3], 2), ' dNLL glu&pres %.4f glu&abs %.4f' % (d_[tem & mk].mean(), d_[tem & glu & ~pres_i].mean()))
```