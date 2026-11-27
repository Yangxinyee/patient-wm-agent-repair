The encoder check failed: the replicated posterior mean correlates only 0.33 with the target on measurement-only bundles and 0.50 on drug-only bundles. With both encoders off by a similar amount, the likelier culprit is shared: how I form the posterior's context or read its output, not a wrong unit row. My head replica only weakly validates ctx, because the prior z enters predictions through a small z_head term.

Next, I will vary the shared pieces on all later bundles:
- context with h before or after the time flow;
- gap scaling log dt or log dt/10, or no gap;
- which half of z_post's 64 outputs is the mean.

The variant that brings both bundle types to correlation near 1 identifies the error. If none does, the encoders themselves are wrong and I'll go back to the first-bundle case that last session matched at 0.976.

```python
Hraw_ = X3[pkb,192:960].reshape(-1,4,192)
variants = {
 'flow_gap10': X3[pkb,:192] + lnorm(Hs_.mean(1), P['ow'], P['ob']) + mlp2(np.log(dtb)[:,None]/10.0, 'gap'),
 'noflow_gap10': X3[pkb,:192] + lnorm(Hraw_.mean(1), P['ow'], P['ob']) + mlp2(np.log(dtb)[:,None]/10.0, 'gap'),
 'flow_gap1': X3[pkb,:192] + lnorm(Hs_.mean(1), P['ow'], P['ob']) + mlp2(np.log(dtb)[:,None], 'gap'),
 'flow_nogap': X3[pkb,:192] + lnorm(Hs_.mean(1), P['ow'], P['ob']),
}
for nmv, cx in variants.items():
    zo_ = mlp2(np.c_[cx, E0, U0], 'z_post')
    for half in [0,1]:
        zz_ = zo_[:, 32*half:32*half+32]
        line = []
        for kk in ['meas','drug']:
            ix = kinds==kk; line.append(f"{kk} corr {np.corrcoef(zz_[ix].ravel(), tgt[ix].ravel())[0,1]:.4f}")
        print(nmv, "half", half, " | ".join(line))
print("target z scale (sd):", tgt.std().round(3), " my zo first-half sd:", zo[:,:32].std().round(3))
```