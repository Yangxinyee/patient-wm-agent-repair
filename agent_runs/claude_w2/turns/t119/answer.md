The apparent discrepancy was in my metric. `zerr_custom` reports the absolute mean error per bundle, while I divided by the mean |z|. The z-posterior means on glucose bundles are small, so a relative error of 0.108 can match an absolute error of 0.013.

Next I recompute my errors in absolute terms, alongside the typical |z| per kind. If they reproduce `zerr_custom`'s numbers (0.0024, 0.002, 0.013, 0.055), my encoder is confirmed equal to my best earlier one. I can then go straight to implementing and verifying the core step, which updates h, out and the memory.

```python
hf_=Hs_[pv]*torch.exp(-dts[:,None,None]*torch.exp(WO['core.flow.a']))
ctx_=OUTs[pv]+LN(hf_.mean(1),WO,'core.out_norm')+mlp(torch.log(dts)[:,None]/10,WO,'gap','gelu')
zm=zpost_mean(ctx_,Eo[nf],Uo[nf],WO)
ab=(zm-Zs[nf]).abs().mean(1).numpy(); mz=Zs[nf].abs().mean(1).numpy()
print('abs err', {k_:round(float(ab[kt==k_].mean()),4) for k_ in np.unique(kt)})
print('mean|z| ', {k_:round(float(mz[kt==k_].mean()),4) for k_ in np.unique(kt)})
print('custom  ', {k_:round(float(r93[kt==k_].mean()),4) for k_ in np.unique(kt)})
```