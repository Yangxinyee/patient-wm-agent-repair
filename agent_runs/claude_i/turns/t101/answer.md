The unit index makes almost no difference (error about 0.055 for every choice), so something else in drug atoms is wrong. Single-START insulin bundles alone have error 0.10, which puts the problem in the START atom's own embedding, not the END atoms. Candidates are the categorical tables I haven't used (action, route, status, text), a different kq index for drugs, or the numeric scaling of rate and dose. Next I'll print the vocabulary's numeric entries for the drug codes, then grid kq and the extra categorical rows on single-atom insulin bundles. An exact match for one combination would identify the missing term.

```python
print([ (k, {kk: (round(v, 3) if isinstance(v, float) else (len(v) if isinstance(v, list) else v)) for kk, v in n_.items() if kk != 'fine_bin_edges'}) for k, n_ in NUMV.items() if k[0] in ('insulin','norepinephrine')])
single = np.zeros(NB, bool)
cnt = Ad.groupby('bi').size(); kin = Ad.groupby('bi').kind.max(); cod = Ad.groupby('bi').code.first()
single[cnt.index[(cnt.values==1) & (kin.values==0)]] = True
so = single & prev_ok.numpy() & (encC['ne'] == 0).numpy()
print('single-start drug bundles', so.sum(), pd.Series(cod.reindex(np.where(so)[0]).values).value_counts().to_dict())
```