The value head is now clear. The logits are `W2·GELU(W1·with_code + b1) + b2`, plus `embed_transition[code*17 + last bin]`. `EDR` holds, for each dose edit, the row indices `gi`, the change in head input `dX`, the dose changes over 4 lags `dd`, and the `reached` flag.

Before building the edit I need the scale of the existing decoders `vI` and `vN`. Specifically:
- the factual noise of `v·Rf` (within-cell) on the target rows;
- the response `v·dX` to each dose edit;
- how that response relates to the dose changes and the impulse responses `hI_full` / `hN_full`.

If the response tracks the true per-row shift with low factual noise, I'll build the hidden-unit edit from it. If the noise is too large relative to the budget, I'll need to rescale or regularise first.

```python
print("hI_full", np.round(hI_full,2), "hN_full", np.round(hN_full,2))
for (drug, fac), dct in EDR.items():
    v_ = vI if drug=='insulin' else vN
    gi_ = dct['gi']; dX_ = dct['dX']; dd_ = dct['dd']; rc = dct['reached'].astype(bool)
    r_ = dX_ @ v_
    hh_k = (hI_full if drug=='insulin' else hN_full)[:4]
    tgt = dd_ @ hh_k
    A1 = np.c_[dd_[rc], np.ones(rc.sum())]
    cf_ = np.linalg.lstsq(A1, r_[rc], rcond=None)[0]
    print(drug, fac, "codes", np.bincount(cf[gi_], minlength=4), "reached", rc.mean().round(3),
          "resp mean", r_[rc].mean().round(3), "tgt4 mean", tgt[rc].mean().round(3),
          "corr", np.corrcoef(r_[rc], tgt[rc])[0,1].round(3), "coef on dd", np.round(cf_,2))
for nm, v_ in [('vI', vI), ('vN', vN)]:
    s_ = Rf @ v_; t_ = Xf @ v_
    print(nm, "within-cell sd by code", [np.round(s_[cf==c].std(),3) for c in range(4)],
          "raw sd by code", [np.round(t_[cf==c].std(),2) for c in range(4)], "|v|", np.linalg.norm(v_).round(3))
```