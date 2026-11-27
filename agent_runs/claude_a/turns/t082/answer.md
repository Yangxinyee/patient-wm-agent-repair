The headline readout is the pooled mean of `shift` over the "reached" stay-hours: 456 for insulin and 305 for norepinephrine on sel1. That matches the summary (−1.879, 0.975, −0.411, −0.520). Forward 4 was only a states check of a trial `z_post` edit, which is no longer applied.

Next I'll record this in the notes and print my current effect kernels (`psi_ins`, `psi_ne`). I'll also compute the readouts those kernels imply on exactly the same 456/305 reached stay-hours of sel1, using my `true_shift` helper. That gives a concrete target to compare with the original model's −1.88 / +0.98 / −0.41 / −0.52, before I re-check the kernel estimates themselves.

```python
with open('/work/notes.md','a') as fh:
    fh.write("- Runs71-77: make_atoms(df,stays)->(atoms,bundles) reproduces AT/SB exactly; encode_bundles(atoms, rows)->(e,u) exact. head_value() batched value head validated (4-sample rmse/NLL = tool's). "
             "dose_edit readout = POOLED mean of shift over 'reached' stay-hours (fwd1 csv). fwd4 = states of a z_post test edit (dropped).\n")
print("psi_ins", np.round(psi_ins,3), "sum", psi_ins.sum()); print("psi_ne", np.round(psi_ne,3), "sum", psi_ne.sum())
print("mult", mult); print("rI type", type(rI), list(rI.items())[:3] if isinstance(rI, dict) else '')
tsh = []
for _, r_ in D1f[D1f.reached].iterrows():
    tsh.append(true_shift(int(r_.stay_id), int(r_.hour), r_.edit))
D1f.loc[D1f.reached, 'true'] = tsh
print(D1f[D1f.reached].groupby('edit')[['shift','true']].mean())
```