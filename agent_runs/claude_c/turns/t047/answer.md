Dimension 26 of z mixes the tier's expected rate (coefficient 1.37) with the innovation (0.65), so it is not innovation-coded. It also varies about as much at reading bundles (sd 1.64) and norepinephrine bundles (sd 1.49) as at insulin bundles. An edit along it would therefore be neither selective nor free of double counting; I drop that route.

The factual metric averages over 4,751 readings: 1,538 finger glucose, 488 lab glucose, 1,109 arterial MAP and 1,616 cuff MAP.

Next I measure how expensive double counting would be. I take the finger-glucose prediction residual sd, overall and after insulin. From it I estimate the NLL cost of a lag-1 bias of −1.98 × expected tier rate at readings after insulin, averaged over all readings. If the cost is well below 0.005, a raw-dose amplification is affordable. If it is near or above 0.005, I need an innovation-type or compensated edit.

```python
fa2['h'] = fa2['hour'].astype(int)
gfz = fa2[fa2.event=='glucose_finger'].copy()
gfz['resid'] = gfz['observed'] - gfz['predicted_mean']
# insulin dose at previous hour (h-1) from Ptu
def lk(col, s, h):
    try: return Ptu.loc[(s, h), col]
    except KeyError: return np.nan
gfz['ins_prev'] = [lk('insulin', s, h-1) for s, h in zip(gfz.stay_id, gfz.h)]
gfz['g_prev'] = [lk('glucose_finger', s, h-1) for s, h in zip(gfz.stay_id, gfz.h)]
gfz['ins_prev'] = gfz['ins_prev'].fillna(0)
gfz['tier_prev'] = base_of(gfz['g_prev'].values)
gfz['exp_prev'] = gfz['tier_prev'].map(lambda t: tier_mean.get(t, 0.0) if t>0 else 0.0)
print("resid sd all", gfz.resid.std().round(2), "after insulin", gfz[gfz.ins_prev>0].resid.std().round(2), "frac after ins", (gfz.ins_prev>0).mean().round(3))
print(gfz.groupby('tier_prev')['resid'].agg(['mean','std','count']).round(2))
sig = gfz[gfz.ins_prev>0].resid.std()
bias = 1.98*gfz['exp_prev']
print("approx NLL increase per reading (all 4751) from lag-1 double counting:", round((bias**2/(2*sig**2)).sum()/len(fa2), 4))
# NLL by event and mean resid
print(fa2.groupby('event')['nll'].mean().round(4))
# within-tier slope of residual on dose innovation (truth ~ -2)
m_ = (gfz.ins_prev>0) & gfz.g_prev.notna()
xx = gfz.loc[m_,'ins_prev'] - gfz.loc[m_,'exp_prev']
print("resid ~ innovation slope:", np.polyfit(xx, gfz.loc[m_,'resid'], 1).round(3), "n", m_.sum())
```