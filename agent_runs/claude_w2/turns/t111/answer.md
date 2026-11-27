The atom embedding now matches exactly. The z-posterior check, though, shows large errors for bundles with readings (relative error about 1.2 for glucose bundles), while action-only bundles are close (0.028). So the context is roughly right, but my encoding of reading bundles differs from the model's.

My earlier, better-verified encodings are still in the kernel (Eb/Ub, E_/U_, E2/U2, Eb_/Ub_). Next I compare them with my new E and U by bundle kind, and I compute the z error with each stored version using my context. If one stored E gives about 0.01 error on glucose bundles, the difference between it and mine shows which encoder detail I've lost. If none does, the problem is in the context rather than the encoder.

```python
def zerr_with(Ex,Ux,gact='gelu'):
    hf_=Hs_[pv]*torch.exp(-dts[:,None,None]*torch.exp(WO['core.flow.a']))
    g_=mlp(torch.log(dts)[:,None]/10,WO,'gap',gact)
    ctx_=OUTs[pv]+LN(hf_.mean(1),WO,'core.out_norm')+g_
    zm=zpost_mean(ctx_,Ex[nf].float(),Ux[nf].float(),WO)
    err=(zm-Zs[nf]).abs().mean(1)/Zs[nf].abs().mean(1)
    return {k_:round(float(err[torch.as_tensor(kt==k_)].mean()),4) for k_ in np.unique(kt)}
for nmE,nmU in [('Eb','Ub'),('E_','U_'),('E2','U2'),('Eb_','Ub_')]:
    Ex=globals()[nmE]; Ux=globals()[nmU]
    dE=(Ex.float()-Eo).abs().mean(1); dU=(Ux.float()-Uo).abs().mean(1)
    print(nmE, 'dE by kind', {k_:round(float(dE[torch.as_tensor(kind_tab[rowB]==k_)].mean()),3) for k_ in np.unique(kt)},
          'dU', {k_:round(float(dU[torch.as_tensor(kind_tab[rowB]==k_)].mean()),3) for k_ in np.unique(kt)})
    print('   zerr', zerr_with(Ex,Ux))
```