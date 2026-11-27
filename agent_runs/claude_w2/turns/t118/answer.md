My E now equals `bundle_enc`'s E exactly for every kind except action bundles (0.034). Yet `zerr_custom` gets 0.013 on finger-glucose bundles where I get 0.108 with identical E. So for those bundles the difference must be in U or in which bundle is indexed.

Next I check three things:
- whether `ip_` equals my index of non-first bundles;
- whether the global `u_empty` and `e_empty` equal my empty-set encodings;
- how `bundle_enc`'s U differs from mine on action bundles.

If `ip_` is offset relative to my indexing, the comparison pairs are not the same and the "better" errors are an artefact. If `u_empty` differs from my empty encoding, that is the missing rule.

```python
ipn=np.asarray(ip_)
print('ip_ == nf ?', len(ipn), len(nf), bool((ipn==nf).all()) if len(ipn)==len(nf) else 'len', ipn[:8], nf[:8], pv[:8])
uemp_g=torch.as_tensor(np.asarray(u_empty)).float().reshape(-1); eemp_g=torch.as_tensor(np.asarray(e_empty)).float().reshape(-1)
myu=U0[~torch.as_tensor(hasu0)][0]; 
isa_b=np.zeros(len(B0),bool)
key0=A0n.stay_id.values.astype(np.int64)*100000+A0n.bidx.values
uk,gid0=np.unique(key0,return_inverse=True)
hasE=np.bincount(gid0[~isa0],minlength=len(uk))>0
mye=E0[torch.as_tensor(~hasE)][0] if (~hasE).any() else None
print('u_empty diff', float((uemp_g-myu).abs().max()), 'e_empty diff', (float((eemp_g-mye).abs().max()) if mye is not None else 'no empty E'))
dU=(torch.as_tensor(np.asarray(Ubc)).float()-Uo).abs().mean(1).numpy()
print('dU', {k_:round(float(dU[kind_tab[rowB]==k_].mean()),4) for k_ in np.unique(kt)})
```