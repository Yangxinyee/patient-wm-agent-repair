Both builds are clean in fp32 on the held-out stays:
- **C = −2.37.** NLL improves by 0.0020 overall. The lag-1 residual slope goes from −1.6 to −0.01. Bias on rows that should be untouched is 0.03 mg/dL on glucose and −0.09 on MAP.
- **C = −4.92.** NLL costs +0.0011 overall (+0.0075 on glucose rows with insulin). It over-corrects lag 1 by +2.1, as designed.

The bf16 check still shows a systematic ±1 mg/dL. It flips sign with C (−0.94 and +1.09). That pattern points to weight rounding, not activation rounding. The gate part of the two paired units' weights, K = κ(w_f + w_p), is shared and probably large. In bf16, K plus the signal and K alone round differently, so their difference is no longer the signal weights.

The fix is twofold:
- make the shared gate weights small, using a stronger ridge on the gates, which still separate cleanly;
- pre-round every edited value to bf16-representable numbers. Then the GPU's conversion is lossless, and fp32 factual scoring and bf16 rollouts see the same weights.

I will compare weight norms and gate quality at ridge penalties 0.01 and 1. Then I will re-run the bf16 check with pre-rounded weights at κ = 6 and κ = 10. Success means a bf16-versus-fp32 bias of 0.2 mg/dL or less, with the fp32 NLL unchanged.

```python
def bfr(a_): return torch.tensor(np.asarray(a_, dtype=np.float32)).to(torch.bfloat16).float().numpy()
for al_ in [0.01, 1.0]:
    wf_a, bf_a = zridge(trm, glu.astype(float), al_); wp_a, bp_a = zridge(trm, pres_i.astype(float), al_)
    Gt = WC0 @ (wf_a + wp_a) + bf_a + bp_a
    print('gate alpha %g: rms(wf+wp) %.3f max %.2f | test G glu&pres pct1 %.2f, glu&abs pct99 %.2f, map&pres pct99 %.2f' % (
        al_, np.sqrt(np.mean((wf_a+wp_a)**2)), np.abs(wf_a+wp_a).max(), np.percentile(Gt[tem&glu&pres_i],1), np.percentile(Gt[tem&glu&~pres_i],99), np.percentile(Gt[tem&mapf&pres_i],99)))
print('rms ws %.3f, cs*ws rms at C=-4.92: %.4f' % (np.sqrt(np.mean(ws**2)), np.sqrt(np.mean((100*-4.92/(4*147)*ws)**2))))
def build_ins2(C, lam=4.0, kap=6.0, units=(31,143), Vref=147.0, wfp=None):
    wf_a, bf_a, wp_a, bp_a = wfp
    W1 = W1o.astype(np.float64).copy(); b1 = b1o.astype(np.float64).copy(); W2 = W2o.astype(np.float64).copy(); b2 = b2o.astype(np.float64).copy()
    ua, ub = units; cs = 100*C/(lam*Vref)
    Kw = bfr(kap*(wf_a + wp_a)); Kb = bfr(kap*(bf_a + bp_a - 1.5))
    W1[ub] = Kw; b1[ub] = Kb
    W1[ua] = bfr(Kw + cs*ws); b1[ua] = bfr(Kb + cs*bs_)
    t_ = bfr(lam*np.r_[cen_gf.mean()/100, cen_gf/100])
    W2[:, ua] = t_; W2[:, ub] = -t_
    return [x_.astype(np.float32) for x_ in (W1, b1, W2, b2)]
for al_ in [0.01, 1.0]:
    wfp_ = zridge(trm, glu.astype(float), al_) + zridge(trm, pres_i.astype(float), al_)
    for kap_ in [6.0, 10.0]:
        E_ = build_ins2(-4.92, kap=kap_, wfp=wfp_)
        nll_e, mean_e = eval_E(*E_); _, mean_e16 = eval_E(*E_, dt=torch.bfloat16)
        d_ = nll_e - nll_o
        print('alpha %g kap %g: test dNLL %.5f | bf16-fp32 bias glu&pres %.3f (sd %.2f), glu&abs %.3f, map %.3f' % (
            al_, kap_, d_[tem].mean(), (mean_e16-mean_e)[tem&mk].mean(), (mean_e16-mean_e)[tem&mk].std(), (mean_e16-mean_e)[tem&glu&~pres_i].mean(), (mean_e16-mean_e)[tem&mapf].mean()))
```