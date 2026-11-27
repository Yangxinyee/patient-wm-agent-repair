With context = h after a time flow in seconds plus gap(log dt/10), the visit-only final bundles reach 0.9525, the best variant. That's close but not exact even for atoms with no values, so something structural is still missing in how the posterior mean is formed.

A common design in sequential latent models is to output the posterior as a correction to the prior: posterior mean = prior mean + z_post(…). This would explain a high but imperfect correlation, because the prior part varies with the state.

Next, I will test that on the final bundles and then on all later bundles. If prior mean plus z_post output matches the target almost exactly, the formula is found, and measurement/drug encoders can be validated cleanly afterwards.

```python
Hf = Hraw_[jl]*np.exp(-(dtb[jl])[:,None]*np.exp(P['fa'])[None,:])[:,None,:]
cxl = X3[pkb[jl],:192] + lnorm(Hf.mean(1), P['ow'], P['ob']) + mlp2((np.log(dtb[jl])/10)[:,None], 'gap')
zol = mlp2(np.c_[cxl, El, np.zeros_like(El)], 'z_post'); zpl = mlp2(cxl, 'z_prior')
for nm_, cand_ in [('post only', zol[:,:32]), ('prior+post', zpl[:,:32]+zol[:,:32]), ('prior only', zpl[:,:32]),
                   ('prior + post*exp(prior_logsd)', zpl[:,:32]+zol[:,:32]*np.exp(zpl[:,32:]))]:
    print(nm_, round(np.corrcoef(cand_.ravel(), tl.ravel())[0,1],4), "MAD", np.abs(cand_-tl).mean().round(4))
# all bundles with default encoders
zpb = mlp2(ctxb, 'z_prior')
for kk in ['meas','drug']:
    ix = kinds==kk
    for nm_, cand_ in [('post', zo_def[ix,:32]), ('prior+post', zpb[ix,:32]+zo_def[ix,:32])]:
        print(kk, nm_, round(np.corrcoef(cand_.ravel(), tgt[ix].ravel())[0,1],4))
```