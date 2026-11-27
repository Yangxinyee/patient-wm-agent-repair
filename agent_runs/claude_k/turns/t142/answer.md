Three points from run 128:
- **`M4`** is the reading table merged with the tool's outputs. It has the register fields I need: `prev_val`, plus `prev_hr`, from which the age can be computed.
- **`last_status`** row 0 is zero and row 1 has norm 12.3, so reading atoms almost certainly get row 0, which adds nothing.
- **Bin 0 rows** of `embed.bins.rate` and `embed.bins.value` are zero, so an absent last-rate bin adds nothing either.

So with_code for a reading should be q + code embedding + last_numeric(8 inputs) + last-value-bin embedding.

I don't know what `BASE99` holds, so I'll test whether it is the non-q, non-last_numeric part. The check is whether WC99 − Q99 − BASE99 − last_numeric.bias lies in the 8-column span of last_numeric.weight. I'll also print the normalisation entries of `vocab.json` so I can compute z_last and the age feature exactly.

If the residual lies in that span, I can swap in the A2 layout cleanly and get with_code that matches `mA2`.

```python
Wln_=L('last_numeric.weight'); bln_=L('last_numeric.bias')
Xold=np.asarray(WC99,float)-np.asarray(Q99,float)-np.asarray(BASE99,float)
for nm_,X_ in [('minus bias',Xold-bln_),('raw',Xold)]:
    coef_,res_,rk_,_=np.linalg.lstsq(Wln_,X_.T,rcond=None)
    r_=X_.T-Wln_@coef_; print(nm_,'resid frac', (r_**2).sum()/(X_**2).sum())
print('old inputs sample (has-prev rows):'); hp=M4.prev_val.notna().values
print(np.round(coef_.T[hp][:3],3)); print(np.round(coef_.T[~hp][:2],3))
print(VOCAB['numeric_note'][:600]); print(json.dumps(VOCAB['numeric'])[:900])
```