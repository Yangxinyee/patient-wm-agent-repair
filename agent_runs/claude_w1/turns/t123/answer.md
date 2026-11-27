My head computation does not match the tool exactly. The mean absolute error is 3.8 mg/dL for finger glucose and 1.4 mmHg for arterial MAP. That is consistent with my earlier note (head replica MAE ~2) and with one known difference: the tool averages probabilities over 4 random prior draws of z, while the vectors use the prior mean. Part of the gap is therefore the tool's own sampling noise.

For the residual regression, exact means are not required. What matters is that my replica's error is not correlated with the dose innovations. I'll test that by regressing (tool − mine) on lagged insulin innovations for the setA finger readings. That needs a policy panel for tuning stays, so I'll write a panel builder for any record set and check it reproduces `Ptr2` on training stays.

```python
def build_panel(df):
    st_ = np.array(sorted(df.stay_id.unique())); ix = {s:i for i,s in enumerate(st_)}; n = len(st_)
    D = np.zeros((n, 26)); Gf = np.full((n, 26), np.nan); Gl = np.full((n, 26), np.nan)
    a = df[df.event.eq('insulin')]; np.add.at(D, (a.stay_id.map(ix).values, np.floor(a.hour.values).astype(int)+1), a.dose.values)
    for ev, Gm_ in [('glucose_finger', Gf), ('glucose_lab', Gl)]:
        b = df[df.event.eq(ev)]; Gm_[b.stay_id.map(ix).values, b.hour.values.astype(int)+1] = b.value.values
    Gany = np.where(np.isfinite(Gf), Gf, Gl)
    pon = np.zeros_like(D); pon[:, 1:] = (D[:, :-1] > 0)
    return st_, ix, D, Gany, pon
st_c = S_ord[:200]; stc, ixc, Dc, Gc, pc = build_panel(tr[tr.stay_id.isin(st_c)])
ref = p2.loc[p2.index.get_level_values(0).isin(st_c)]
print('D match', np.allclose(Dc.ravel(), ref.D.fillna(0).values), '| Gany match', np.allclose(np.nan_to_num(Gc.ravel(), nan=-1), ref.Gany.fillna(-1).values), '| prev_on match', np.allclose(pc.ravel(), ref.prev_on.values))
```