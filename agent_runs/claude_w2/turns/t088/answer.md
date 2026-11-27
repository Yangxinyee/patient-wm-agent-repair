The bundles line up with the tool's states. For example, stay 1010002352 has an action bundle at hour 0.0833 with 1 atom. The bundle at 1.0833 has 2 atoms: a new norepinephrine record plus the END atom of the previous one. So END atoms are part of the bundles and must be included. The states file gives per bundle: `x` (out, 4 slots of h, z posterior mean; 992 numbers), stay_id, hour, has_insulin, has_norepinephrine and n_atoms.

**Next step.** My old `zpred` helper should reproduce the z posterior mean at a bundle from the previous state and the bundle's atoms. That runs through the atom embedding, both set encoders, ctx and `z_post`. Its default arguments (gelu, unit_row=2, side='left') are presumably the variant I found best before the restart.

This run does two checks:
- whether `stay_atoms` produces the same atom count per bundle as the tool, including END atoms, across one stay;
- how closely `zpred` matches the stored z on a set of that stay's bundles.

If z matches to about 1e-4, the input side is verified and only the prelude, blocks and coda remain to write. A poor match means the embedding details (units, bins or END handling) still need fixing.

```python
sidS_ = S['stay_id']; hS_ = S['hour']; xS = S['x']
sid_ = int(sidS_[0])
A_ = stay_atoms(trs.loc[[sid_]])
print(A_[A_.kind == 2].head(4).to_string())
idx_ = np.where(sidS_ == sid_)[0]
cnt_mine = A_.groupby('minute').size()
mins_tool = np.round(hS_[idx_] * 60).astype(int)
print('n bundles tool', len(idx_), 'mine', len(cnt_mine), 'atom counts equal:', np.array_equal(cnt_mine.reindex(mins_tool).fillna(-1).values.astype(int), S['n_atoms'][idx_].astype(int)))
errs_z = []
for j_ in range(1, min(len(idx_), 30)):
    ik_, ip_ = idx_[j_], idx_[j_ - 1]
    a_ = A_[A_.minute == mins_tool[j_]]
    dt_s = (hS_[ik_] - hS_[ip_]) * 3600
    try:
        zp_ = zpred(ip_, ik_, a_, dt_s)
        zp_ = zp_.numpy() if hasattr(zp_, 'numpy') else np.asarray(zp_)
        errs_z.append(np.abs(zp_.ravel()[:32] - xS[ik_, 960:992]).max())
    except Exception as ex:
        print('zpred err', j_, ex); break
print('z max abs err over bundles:', np.round(errs_z, 5))
print('z scale', np.abs(xS[idx_, 960:992]).mean())
```