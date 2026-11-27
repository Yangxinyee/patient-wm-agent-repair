The readouts are mixed:
- pibar(g) from ctx: R² 0.94, usable;
- insulin rate from u: R² 0.83 linearly;
- NE rate from u: R² 0.60.

The u readouts are limited because u is a LayerNorm'd nonlinear set encoding. A more accurate rate readout inside z_post would need several repurposed hidden units, so the cost of repurposing matters.

I'll rank z_post's 192 hidden units by how much the posterior mean would move if a unit were frozen at its mean, over all 5156 bundles. I'll also count near-dead units (activation spread under 0.01). Several units with negligible importance would make a multi-unit, gated deviation readout cheap. If all units matter, I need another injection site.

```python
zz3, Hz = zpost2(ctx2, torch.tensor(Eall), torch.tensor(Uall))
Hz = Hz.numpy(); W2z = T64('z_post.2.weight').numpy()[:32]
dev_ = Hz - Hz.mean(0)
imp = np.sqrt(((dev_[:,:,None]*W2z.T[None]).reshape(len(Hz),192,32)**2).sum(2).mean(0))
zsd = np.sqrt(((zz3[:,:32].numpy()-zz3[:,:32].numpy().mean(0))**2).sum(1).mean())
o_ = np.argsort(imp)
print('total z rms dev', zsd.round(3))
print('least important units', o_[:12], imp[o_[:12]].round(4))
print('activation sd of those', Hz[:,o_[:12]].std(0).round(4), 'mean', Hz[:,o_[:12]].mean(0).round(3))
print('importance quantiles', np.quantile(imp,[0,0.05,0.1,0.25,0.5,0.9,1]).round(4))
print('near-dead (act sd<0.01):', int((Hz.std(0)<0.01).sum()))
```