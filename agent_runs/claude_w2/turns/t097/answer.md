Two more details are settled: END atoms carry no rate or dose, and bins use side='left'. With these the action-only error falls from 0.045 to 0.017, and reading-only bundles are at 0.019.

Glucose and MAP use the unit rows in vocab code order (mmHg = 2, mg/dL = 3). Alphabetical order or a swap is clearly worse. The drug unit rows and the rate_unit order barely matter.

At about 10% of the z scale, though, the remaining error is too big to be float noise. So something small but systematic is still off.

**This run.** Under the new best setting I re-test the remaining small options:
- the drug unit order;
- the rate_unit order;
- kq;
- tanh-approximated GELU;
- LayerNorm eps.

I also break the error down by bundle composition (which codes are present, atom count) and look at its distribution. If the error concentrates in one bundle type, such as glucose_lab readings or visit atoms, I know which atom's embedding is still wrong. A uniform spread would instead suggest a numerical convention (GELU form or eps).

```python
def z_check4(umap=U_codes, rumap=RU1, use_kq=True, gelu_tanh=False, eps=1e-5, return_err=False):
    global ACT
    ACT['gelu'] = (lambda x: F.gelu(x, approximate='tanh')) if gelu_tanh else F.gelu
    A2_ = ATM.copy(); A2_.loc[A2_.kind == 2, ['rate', 'dose']] = np.nan
    num, vb, db, rb = atom_feats(A2_, 'left')
    xa = embed_atoms(A2_, num, vb, db, rb, umap, rumap, use_kq)
    Eb, Ub = bundle_enc(xa, A2_.bidx.values, A2_.ekind.values == 2, nB, 'gelu')
    ee = empty_enc('encode'); ue = empty_enc('encode_action')
    Eb[~torch.from_numpy(hasmeas)] = ee; Ub[~torch.from_numpy(hasact)] = ue
    on_ = LNt(hf.mean(1), 'core.out_norm', eps); g2 = mlp2l((torch.log(dt_) / 10)[:, None], 'gap', 'gelu')
    zm_ = zpost_from(out_p + on_ + g2, Eb[ip_], Ub[ip_])
    ACT['gelu'] = F.gelu
    e_all = (zm_ - ztrue).abs().mean(1)
    if return_err: return e_all.numpy(), Eb, Ub
    return float(e_all[~act_m].mean()), float(e_all[act_m].mean())
for nm_, kw in [('base', {}), ('c3', dict(umap=U_c3)), ('ru2', dict(rumap=RU2)), ('ru0', dict(rumap={})), ('nokq', dict(use_kq=False)), ('tanh', dict(gelu_tanh=True)), ('eps1e-6', dict(eps=1e-6))]:
    print(nm_, np.round(z_check4(**kw), 4))
eb_, Eb_, Ub_ = z_check4(return_err=True)
comp = ATM.groupby('bidx').apply(lambda d: ','.join(sorted(set(CODEN[int(c)][:6] + ('E' if k == 2 else '') for c, k in zip(d.code, d.kind)))))
cb = comp.reindex(ip_).values
dfE = pd.DataFrame({'err': eb_, 'comp': cb})
print(dfE.groupby('comp')['err'].agg(['count', 'mean', 'median']).sort_values('count', ascending=False).head(14).round(4).to_string())
print('quantiles', np.round(np.quantile(eb_, [.1, .5, .9, .99]), 4))
```