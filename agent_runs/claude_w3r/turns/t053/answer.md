Zeroing e_k triples the error on observation bundles, and zeroing u_k quadruples it on action bundles. So both encoders contribute and are roughly right. Dropping kq changes almost nothing. The residual of about 0.04–0.05 is therefore probably in ctx, and the most direct check is the stated identity out = core.out_norm(mean_slots(h)).

If the stored h is the post-coda y, that identity should hold exactly in the states. If it fails, the stored h is a different quantity, for example pre-coda, and my ctx would be built from the wrong h.

```python
Xall = torch.tensor(S_['x'])
out_all = Xall[:,:192]; h_all = Xall[:,192:960].reshape(-1,4,192)
recon = LN(h_all.mean(1),'core.out_norm')
print('out vs out_norm(mean h): mean abs err', float((recon-out_all).abs().mean()), 'mean |out|', float(out_all.abs().mean()))
# alternative: maybe out_norm uses c_out(h) or no affine
recon2 = Fnn.layer_norm(h_all.mean(1),(192,))
print('no-affine LN err', float((recon2-out_all).abs().mean()))
print('corr per-dim median', float(np.median([np.corrcoef(recon[:,j],out_all[:,j])[0,1] for j in range(192)])))
```