The states pass returned original and edited states on the same 100 stays. Next I measure each model's dose sensitivity at insulin bundles. I regress the change across the insulin bundle (z, out, h) on the dose innovation, the tier's expected rate, glucose and the presence of norepinephrine, then compare the innovation coefficients between the models.

An edited/original ratio near 5 for out and h means the state gets the amplified signal and the readout bottleneck is downstream. A ratio near 1.4 means the action encoder saturates. I also save my recent findings to the notebook first.

```python
notes_add_txt = """
## Restart-2 findings (runs 37-47, pass 5)
- ops 1-2 = rank1 'in' on embed.numeric.weight along e2 (z_dose) and e4 (z_rate), alpha=4 (cols x5). Pass 4: ins x2 -2.51, half +1.66, ne x2 +0.14, half -0.02.
- Per 'since' (hrs since first edited dose), model/true ratio ~0.15 at all lags for insulin x2 (lag1 -0.43 vs -2.90; lag12 -3.37 vs -19.55): time shape OK, per-step gain too small.
- NE: original's negative-for-both comes from cap stays (rate 1.0); at cap half gives -0.7 (true -1.69 lag1); x2 at cap -0.5..-1.0 (OOD: rate 2.0 -> bin16, z 2.6 > train max 0.92). ops1-2 flattened NE responses to ~0.
- z posterior: at insulin bundles only z dim 26 varies (var 2.28); z26 ~ 0.65*innov + 1.37*exp_rate - 0.31*g/50; z26 also varies at reading bundles (sd 1.6) -> not selective.
- Factual-cost estimate (Gaussian, sigma_gf=14.5): lag-1 effect -k*(dose-d0): d0=0,k=1.98 -> +0.0072/reading; d0=1.5 (z-score centre) -> +0.0023. Centering at median makes raw-dose amplification affordable.
- Original model residual ~ innovation slope -1.37 at finger readings after insulin (model under-uses dose).
- ids100 = tuning stays 2020000000..2020000099.
"""
with open('/work/notes.md','a') as f: f.write(notes_add_txt)

So = np.load('/work/fwd_5/states_original.npz'); Se = np.load('/work/fwd_5/states_edited.npz')
assert (So['stay_id']==S3['stay_id']).all()
def dose_sens(Sx):
    x = Sx['x']; sidx = Sx['stay_id']
    prev = np.r_[-1, np.arange(len(x)-1)]
    same = np.r_[False, sidx[1:]==sidx[:-1]]
    rows_ = ins_b[same[ins_b]]
    sel = np.isin(ins_b, rows_) & ok
    rr = ins_b[sel]
    D_ = x[rr] - x[rr-1]
    A_ = np.column_stack([np.ones(sel.sum()), innov[sel], exp_rate[sel], (gsrc_b[sel]-200)/50, hn3[rr].astype(float)])
    c_, *_ = np.linalg.lstsq(A_, D_, rcond=None)
    res = D_ - A_ @ c_
    out = {}
    for nm_, sl in [('out', slice(0,192)), ('h', slice(192,960)), ('z', slice(960,992)), ('z26', slice(960+26, 961+26))]:
        out[nm_] = (np.linalg.norm(c_[1, sl]), np.linalg.norm(c_[2, sl]), np.sqrt((res[:, sl]**2).sum(1).mean()))
    return out, c_
so_, co_ = dose_sens(So); se_, ce_ = dose_sens(Se)
for k_ in so_:
    print(k_, "orig |b_innov| %.3f |b_exp| %.3f resid %.3f" % so_[k_], "|| edited |b_innov| %.3f |b_exp| %.3f resid %.3f" % se_[k_],
          " ratio innov %.2f" % (se_[k_][0]/so_[k_][0]))
# cosine between original and edited innovation directions
for nm_, sl in [('out', slice(0,192)), ('h', slice(192,960))]:
    a_, b_ = co_[1, sl], ce_[1, sl]
    print(nm_, "cos(innov dir orig, edited) %.3f" % (a_@b_/np.linalg.norm(a_)/np.linalg.norm(b_)))
```