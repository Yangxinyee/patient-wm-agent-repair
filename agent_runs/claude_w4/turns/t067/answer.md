The probe ran: 5,218 bundles across 100 stays, with z posterior means in columns 960–991 of the state array.

Now I'll read the results:
- **Activation:** z dims 0 and 1. I expect (0.5, 0.0) if ReLU.
- **Block order:** whether the u_k copies (dims 12–31) are exactly 0 at bundles without drugs and nonzero at drug bundles. That would confirm the [ctx, e_k, u_k] input order.
- **Dependence on rate:** for insulin-only and NE-only bundles, how the u_k coordinates vary with the recorded rate. I'll join the bundles to the tuning records by stay and hour.

That tells me whether u_k is a clean, smooth function of rate, which the injection design needs. If the u copies vary at non-drug bundles, the block order is different from what I assumed.

```python
S4 = np.load('/work/fwd_4/states_edited.npz')
X4 = S4['x']; Z4 = X4[:, 960:992]
sid4, hr4, hi4, hn4, na4 = S4['stay_id'], S4['hour'], S4['has_insulin'].astype(bool), S4['has_norepinephrine'].astype(bool), S4['n_atoms']
print('act test dims (unique):', np.unique(np.round(Z4[:,0],4))[:5], np.unique(np.round(Z4[:,1],4))[:5])
nod = ~(hi4|hn4)
ucols = [j for (j,b,c) in probe_cols if b=='u']
print('u-copies |max| at non-drug bundles:', np.abs(Z4[nod][:,ucols]).max().round(4), ' at drug bundles mean|.|:', np.abs(Z4[~nod][:,ucols]).mean().round(3))
print('n bundles: nondrug', nod.sum(), 'ins only', (hi4&~hn4).sum(), 'ne only', (hn4&~hi4).sum(), 'both', (hi4&hn4).sum())
# match drug bundles to records
tun_d = tu[tu.event.isin(['insulin','norepinephrine'])][['stay_id','hour','event','rate']]
ins_r = tun_d[tun_d.event=='insulin'].set_index(['stay_id','hour']).rate
key4 = list(zip(sid4, np.round(hr4,4)))
ins_r.index = [(a, round(b,4)) for a,b in ins_r.index]
ri = np.array([ins_r.get(k, np.nan) for k in key4])
m_io = hi4 & ~hn4
print('ins-only bundles with matched rate:', np.isfinite(ri[m_io]).sum(), 'n_atoms dist', np.unique(na4[m_io], return_counts=True))
U = Z4[:, ucols]
for rv in [0.5,1,2,3,4,6]:
    mm = m_io & np.isclose(ri, rv)
    if mm.sum(): print(rv, mm.sum(), 'u[:6] mean', U[mm][:, :6].mean(0).round(3), 'sd', U[mm][:, :6].std(0).round(3))
```