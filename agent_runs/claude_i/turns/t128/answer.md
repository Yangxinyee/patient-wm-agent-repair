Solved for the end-of-stay bundles. They hold the hospital-admission and ICU-admission END records, i.e. discharge. My stored e for them was wrong (it differs from the correct one by norm 16). Encoding them with the generic END-atom vector (code + kind 2 + event kind + kq 0 + numeric bias) and the sum/√n set rule gives exactly zero h error.

So the core is right, and the remaining errors come from my stored encodings. That is likely also true for the two-drug bundles at h+5min, where two END records share the measurement set.

Next I will look at the atoms of a failing two-drug bundle and recompute its e from the END atoms with the corrected encoder, keeping the stored u.
- **If the error vanishes:** the stored e for multi-END sets was the bug.
- **If it does not:** the stored u for two STARTs is the bug, and I will rebuild the START-atom encoding (z-scores and bins) as well.

I also print the signatures of my z-score and bin helpers for that case.

```python
for fn_name in ['zsc','fbin','zs_','fb_','atoms_embed_x','u_only_x','u_from_V']:
    try: print(fn_name, inspect.signature(globals()[fn_name]))
    except Exception as ex: print(fn_name, 'ERR', type(ex).__name__)
bb2 = [k_ for k_ in bad_ if (z3['has_insulin'][k_]==1) and (z3['has_norepinephrine'][k_]==1)][:4]
for k_ in bb2:
    key_ = (int(S3[k_]), int(round(float(min_[k_]))))
    at_ = atoms_by_bundle[key_]
    ends = [a['code'] for a in at_ if a['kind'] == 2]; others = [a for a in at_ if a['kind'] != 2 and a['code'] not in ('insulin','norepinephrine')]
    print(key_, [(a['code'][:5], a['kind']) for a in at_], 'n_other_meas', len(others))
    if len(others) == 0 and len(ends) > 0:
        e_n = set_enc(torch.stack([atom_end(c) for c in ends]), 'encode')
        print('   stored-e diff', round((e_n - Ee[k_]).norm().item(),3), 'h err stored', round(ER[k_],4), 'h err new e', round(h_err_with(k_, e_new=e_n),4))
```