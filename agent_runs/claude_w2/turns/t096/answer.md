Encoding empty sets as "the empty-set encoding" rather than zero fixes most of the gap, for both e and u, with the full ctx (gap and time flow). This is despite the documentation's "u is zero if the bundle has none".

| Bundle type | Error before | Error now |
|---|---|---|
| Reading-only | 0.134 | 0.018 |
| Action-only | 0.070 | 0.045 |
| Overall | — | 0.030 (corr 0.845) |

So ctx was right all along, and the empty-set handling was the structural bug.

What remains is mostly on action bundles, so it is probably in how drug atoms are embedded. Tied bin sides matter a lot for rates on the 0.5 grid. The other candidates are the unit and rate_unit rows for drugs, and whether END atoms carry the rate and dose of the record that ends.

**This run.** I re-run the embedding grid under the corrected empty handling, reporting the reading-only and action-only errors separately. The extra END variant strips the rate and dose from END atoms. One combination should bring the action-only error down to the reading-only level or below. If none does, the remaining issue is in the action encoder's pooling.

```python
def z_check3(side='left', umap=U_codes, rumap=RU1, use_kq=True, end_attrs=True, ctx_mode='full'):
    A2_ = ATM.copy()
    if not end_attrs:
        A2_.loc[A2_.kind == 2, ['rate', 'dose']] = np.nan
    num, vb, db, rb = atom_feats(A2_, side)
    xa = embed_atoms(A2_, num, vb, db, rb, umap, rumap, use_kq)
    Eb, Ub = bundle_enc(xa, A2_.bidx.values, A2_.ekind.values == 2, nB, 'gelu')
    Eb[~torch.from_numpy(hasmeas)] = e_empty; Ub[~torch.from_numpy(hasact)] = u_empty
    ctx_ = out_p + on_hf + gfull
    zm_ = zpost_from(ctx_, Eb[ip_], Ub[ip_])
    e_all = (zm_ - ztrue).abs().mean(1)
    return float(e_all[~act_m].mean()), float(e_all[act_m].mean()), Eb, Ub
U_c2 = {'map_cuff': 3, 'glucose_finger': 2, 'map_arterial': 3, 'glucose_lab': 2, 'insulin': 4, 'norepinephrine': 5}
U_c3 = {'map_cuff': 2, 'glucose_finger': 3, 'map_arterial': 2, 'glucose_lab': 3, 'insulin': 5, 'norepinephrine': 4}
for side_, (unm, um), (rnm, ru), kq_, ea_ in itertools.product(['left', 'right'], [('codes', U_codes), ('c2', U_c2), ('c3', U_c3), ('alpha', U_alpha)], [('ru1', RU1), ('ru2', RU2), ('ru0', {})], [True, False], [True, False]):
    if (unm != 'codes' or rnm != 'ru1') and not (kq_ and ea_): continue
    m_, a_, _, _ = z_check3(side_, um, ru, kq_, ea_)
    print(f'{side_:5s} {unm:5s} {rnm} kq={int(kq_)} endattr={int(ea_)}  meas {m_:.4f}  act {a_:.4f}')
```