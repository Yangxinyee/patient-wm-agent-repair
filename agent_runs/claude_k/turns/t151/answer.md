Key facts from the config: 4 attention heads of 48 dims, one prelude layer, two recurrent layers and one coda layer. The z latent has 32 dims and is sampled 4 times at evaluation. My helper functions build drug atoms (with a dose multiplier option), set-encode them, and rebuild the context and the posterior mean of z. Their internals are hidden, but their inputs are clear.

The cleanest channel design would use a "dead" z dimension: one whose posterior mean barely varies in training data.
- I could repurpose one z_post hidden unit to write the dose residual into that dimension. It can compute the residual itself, because z_post sees both the state context and u_k.
- I could also set that dimension's z_in column freely, since the model never learned to use it. That gives a controllable injection vector into the prelude.
- Heads use prior draws of z, not the posterior, so the readout would come through the state as before.

This only works if such a dimension exists and the slots pay steady attention to the z token. So first I'll check per-dimension posterior-mean variation across all bundles in the 99-stay state file, against the prior's mean and sd at readings and the norms of the z_in and z_head columns.

- One or more dimensions with posterior sd near 0 would confirm the dead-dimension option.
- If all 32 dimensions are active, I'll fall back to injecting through the u-token value path.

```python
Xs2=S2['x']; zpost=Xs2[:,960:992]
zin_W=L('z_in.weight'); zh_W=L('z_head.weight')
pm_=zp_[:,:32]; psd_=np.exp(zp_[:,32:])
tab_=pd.DataFrame({'post_mean':zpost.mean(0),'post_sd':zpost.std(0),'prior_mean_avg':pm_.mean(0),'prior_sd_avg':psd_.mean(0),
                   'zin_norm':np.linalg.norm(zin_W,axis=0),'zhead_norm':np.linalg.norm(zh_W,axis=0)})
print(tab_.round(3).sort_values('post_sd').to_string())
hi_=S2['has_insulin'].astype(bool); print('post sd on insulin bundles (lowest 6 dims):', np.round(np.sort(zpost[hi_].std(0))[:6],3))
```