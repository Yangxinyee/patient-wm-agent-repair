The error breakdown localises what is left.
- **Exact (0.0000):** MAP-only bundles and the end-of-stay visit END bundles. So the MAP embedding, the encoders and ctx are all right.
- **Small but real error:** bundles that contain glucose (median 0.005–0.008). So something about the glucose atoms is slightly off, most likely their unit row.
- **Large error (0.217):** bundles where infusions end and nothing new starts (insulin END + norepinephrine END only). These bundles are handled differently from what I assumed.

**This run.**
- I test glucose unit rows 2–5, keeping MAP on row 2 and permuting the drug rows. I report the error separately for glucose_finger and glucose_lab bundles.
- For END-only bundles I test three alternatives: END atoms going through the measurement encoder, u set to exactly zero, and END atoms keeping their rate and dose.

A variant that zeroes the glucose-bundle error, plus one that fixes END-only bundles, would complete the input side. If glucose stays slightly off under every unit row, the issue is instead in the glucose value features (z-score or bins).

```python
grp_g = ATM.groupby('bidx')
has_gf = np.zeros(nB, bool); has_gf[ATM.bidx.values[ATM.code.values == 6]] = True
has_gl = np.zeros(nB, bool); has_gl[ATM.bidx.values[ATM.code.values == 10]] = True
endonly = np.zeros(nB, bool)
kinds_b = ATM.groupby('bidx').apply(lambda d: bool(((d.ekind == 2) & (d.kind == 2)).all()))
endonly[kinds_b.index.values[kinds_b.values]] = True
def zerr_custom(umap, rumap=RU1, end_mode='u_noattr'):
    A2_ = ATM.copy()
    if end_mode != 'u_attr': A2_.loc[A2_.kind == 2, ['rate', 'dose']] = np.nan
    num, vb, db, rb = atom_feats(A2_, 'left')
    xa = embed_atoms(A2_, num, vb, db, rb, umap, rumap, True)
    isact = (A2_.ekind.values == 2)
    if end_mode == 'e_route': isact = isact & (A2_.kind.values != 2)
    Eb, Ub = bundle_enc(xa, A2_.bidx.values, isact, nB, 'gelu')
    hm = np.zeros(nB, bool); hm[A2_.bidx.values[~isact]] = True; ha = np.zeros(nB, bool); ha[A2_.bidx.values[isact]] = True
    Eb[~torch.from_numpy(hm)] = e_empty; Ub[~torch.from_numpy(ha)] = u_empty
    if end_mode == 'u_zero_endonly': Ub[torch.from_numpy(endonly)] = 0.0
    zm_ = zpost_from(out_p + on_hf + gfull, Eb[ip_], Ub[ip_])
    e_all = (zm_ - ztrue).abs().mean(1).numpy()
    return e_all
for gr in [3, 4, 5]:
    others = [r_ for r_ in [3, 4, 5] if r_ != gr]
    for perm in [others, others[::-1]]:
        um = {'map_cuff': 2, 'map_arterial': 2, 'glucose_finger': gr, 'glucose_lab': gr, 'insulin': perm[0], 'norepinephrine': perm[1]}
        e_ = zerr_custom(um)
        print(f'glu row {gr} ins {perm[0]} ne {perm[1]}: gf-bundles {e_[has_gf[ip_]].mean():.4f} gl-bundles {e_[has_gl[ip_]].mean():.4f} action {e_[act_m.numpy()].mean():.4f} endonly {e_[endonly[ip_]].mean():.4f}')
for em in ['e_route', 'u_zero_endonly', 'u_attr']:
    e_ = zerr_custom(U_codes, RU1, em)
    print(em, f'gf {e_[has_gf[ip_]].mean():.4f} action {e_[act_m.numpy()].mean():.4f} endonly {e_[endonly[ip_]].mean():.4f} all {e_.mean():.4f}')
```