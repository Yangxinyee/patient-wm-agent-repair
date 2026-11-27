The z posterior strongly encodes reading news. A linear readout of z recovers the reading surprise with high R²:

| Reading | R² | Main z dims |
|---|---|---|
| finger glucose | 0.85 | 20, 27, 9, 30 |
| lab glucose | 0.73 | 20, 7, 9 |
| arterial MAP | 0.79 | 13, 22, 2, 23 |
| cuff MAP | 0.77 | 13, 2, 23, 22 |

So z is the model's own "news" channel into the state, and glucose news and MAP news live in largely separate directions.

That gives a natural design. At insulin bundles, repurposed z_post units push z along the "glucose lower than expected" direction, in proportion to the dose surprise (rate minus the expected rate for the current state). NE bundles get the same treatment with the "MAP higher" direction. The model's own filtering and the fed-back readings should then carry the effect forward.

To build those units I need z_post's exact inputs at drug bundles. ctx and u_k are already exact. e_k there should be the empty-set or END-atom encoding, where an END atom marks a record that began exactly one hour earlier. Next I'll reconstruct the END set for each drug bundle from the records and check it against the probe's exact e_k coordinates in fwd_4. A match on every drug bundle means the z_post inputs are fully known wherever the edit acts. Mismatches would show the END timing rule is different, for example for records starting at h+0.1667.

```python
drec = tu[tu.event.isin(['insulin','norepinephrine'])][['stay_id','hour','event','rate','end_hour']].copy()
drec['hk'] = drec.hour.round(4); drec['ek'] = drec.end_hour.round(4)
starts = drec.groupby(['stay_id','hk']).event.apply(list).to_dict()
ends = drec.groupby(['stay_id','ek']).event.apply(list).to_dict()
def end_atom(code_name): return E_['code'][cid[code_name]] + E_['kind'][2] + E_['ekind'][2] + E_['kq'][0] + NB
def e_drug_bundle(s_, h_):
    en = ends.get((s_, h_), [])
    return enc_e([end_atom(c_) for c_ in en]) if en else enc_e_empty, en
enc_e_empty = enc_set('encode', torch.zeros(0,192))
def enc_e(atoms):
    A = torch.stack(atoms); n = len(atoms)
    h1 = gelu(A @ load('encode.mlp1.0.weight').T + load('encode.mlp1.0.bias')) @ load('encode.mlp1.2.weight').T + load('encode.mlp1.2.bias')
    s = h1.sum(0)/np.sqrt(n) + load('encode.size.weight')[:,0]*np.log1p(n) + load('encode.size.bias')
    h2 = gelu(load('encode.mlp2.0.weight') @ s + load('encode.mlp2.0.bias')) @ load('encode.mlp2.2.weight').T + load('encode.mlp2.2.bias')
    return LNf(h2, load('encode.norm.weight'), load('encode.norm.bias'))
drug_rows4 = np.where(hi4 | hn4)[0]
errs = []; cnt = collections.Counter()
for i in drug_rows4:
    e_, en = e_drug_bundle(sid4[i], round(float(hr4[i]),4)); cnt[tuple(sorted(en))] += 1
    errs.append(float((e_[e_c] - torch.tensor(Z4[i, e_dims])).abs().max()))
errs = np.array(errs); print('drug bundles', len(errs), 'max err', errs.max().round(5), 'n err>1e-3', (errs>1e-3).sum(), 'END sets', dict(cnt))
print('start time fractions:', drec.hour.mod(1).round(4).value_counts().to_dict())
```