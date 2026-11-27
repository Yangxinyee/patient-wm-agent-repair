Many value-head hidden units are almost redundant. The best are explained by the other units with R² 0.9995 (units 121, 92, 130, 141, 191), and the median unit has R² 0.99. Freeing a few units by folding their output into the others' W2 weights should cost little factual accuracy.

That makes a gated design feasible. Each drug gets two repurposed units whose difference outputs gate × decoded surprise. W2 then maps that difference onto a mean-shifting tilt of the right code family's logits.

The gates must be sharp and linearly decodable from `with_code`. For insulin the gate is "glucose code and insulin given in the preceding bundle". For NE I'm considering either "MAP code" or "MAP code and NE running". Next I'll check how well a linear readout separates these groups. Near-perfect separation (worst-case margins in the right order) would confirm the gating plan. Overlapping groups would mean folding the gate into the surprise decoder instead.

```python
gl_all = RD2.event.str.startswith('glucose').values
insA = gl_all & (RD2['ins_1'].fillna(0).values>0)
neA = (~gl_all) & (RD2['ne_1'].fillna(0).values>0)
def lin_sep(y, mask=None):
    m_ = np.ones(len(y),bool) if mask is None else mask
    Xw = np.column_stack([WC0[m_], np.ones(m_.sum())]); yy = y[m_].astype(float)
    cf_ = np.linalg.lstsq(Xw, yy, rcond=None)[0]; pr_ = Xw@cf_
    pos, neg = pr_[yy>0.5], pr_[yy<0.5]
    return dict(n=int(m_.sum()), pos_min=round(pos.min(),3), pos_q01=round(np.quantile(pos,.01),3), neg_max=round(neg.max(),3), neg_q99=round(np.quantile(neg,.99),3), r2=round(1-((yy-pr_)**2).mean()/yy.var(),3))
print('glucose vs map', lin_sep(gl_all))
print('ins gate among glucose', lin_sep(insA, gl_all))
print('ins gate (glucose&ins) among all', lin_sep(insA))
print('ne gate among map', lin_sep(neA, ~gl_all))
```