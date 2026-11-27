Lower ridge penalties help the insulin decoder: at alpha 0.01 the cross-validated R² is 0.62, with slope 0.66. The norepinephrine side is poor:
- Its shortfall coefficients are collinear: 3.15, 0.53, -2.13 on rate deviations. That is because norepinephrine titration makes the rate deviation persist from hour to hour.
- The best norepinephrine decoder reaches only R² 0.21, slope 0.27.

An unbiased norepinephrine channel would therefore inject about 1.4 mmHg of noise. That would cost about 0.009 nats overall, which is over the whole budget, so norepinephrine has to wait.

Insulin is the main readout, so I will now build a bf16-safe insulin-only channel (E3-ins) and test it cleanly:
- **Units.** Reuse the two low-importance hidden units 31 and 143. Ablating them cost only 0.0003.
- **Gates.** Glucose-family and insulin-presence gates, fitted by ridge regression on z-sampled with_code.
- **Pre-activations.** κ = 8 times (g_f + g_p − 1.5), so an open gate sits at about +4 and a closed one at about −4 or −12. The dose signal is added to one unit of the pair.
- **Output.** Columns ±λ·t with t = glucose centres / 100 and λ = 4.

This run fits the gates and decoder on the 300 selT stays only, so the selS stays can be scored honestly. It also checks three things:
- the gate margins;
- whether the presence gate depends on the insulin rate. If it does, halving the dose could close it, which would bring back the lopsided response.
- the decoder's held-out R² and slope.

```python
selS_ = json.load(open('/work/selS.json')); selT_ = json.load(open('/work/selT.json'))
sS = set(int(x) for x in (selS_[list(selS_)[0]] if isinstance(selS_, dict) else selS_))
sT = set(int(x) for x in (selT_[list(selT_)[0]] if isinstance(selT_, dict) else selT_))
trm = np.array([s_ in sT for s_ in sidv]); tem = np.array([s_ in sS for s_ in sidv])
print('train rows', trm.sum(), 'test rows', tem.sum())
def zridge(mask, y, alpha):
    m_ = Ridge(alpha=alpha).fit(XA[:, mask].reshape(-1,192), np.tile(y[mask], ND))
    return m_.coef_.astype(np.float64), float(m_.intercept_)
ins1 = rd_.ins_1.fillna(0).values
wf, bf_ = zridge(trm, glu.astype(float), 0.01)
wp, bp_ = zridge(trm, pres_i.astype(float), 0.01)
ws, bs_ = zridge(trm & mk, e1, 0.01)
gf_ = WC0 @ wf + bf_; gp_ = WC0 @ wp + bp_; sh_ = WC0 @ ws + bs_
G = gf_ + gp_
for nm_, m_ in [('glu&pres', glu&pres_i), ('glu&abs', glu&~pres_i), ('map&pres', mapf&pres_i), ('map&abs', mapf&~pres_i)]:
    print(nm_, 'test G pct1/50/99: %s' % np.round(np.percentile(G[tem&m_], [1,50,99]),2))
mm_ = tem & mk
print('presence gate vs ins rate (test present glu rows): corr %.2f; slope per u/h %.3f' % (np.corrcoef(gp_[mm_], ins1[mm_])[0,1], np.polyfit(ins1[mm_], gp_[mm_], 1)[0]))
r2s = 1 - np.mean((sh_[mm_]-e1[mm_])**2)/e1[mm_].var(); ks = np.polyfit(e1[mm_], sh_[mm_], 1)[0]
print('decoder test R2 %.3f slope %.3f; resp to ins rate at fixed exp: %.3f' % (r2s, ks, np.linalg.lstsq(np.c_[ins1[mm_], np.nan_to_num(rd_.ins_exp_1.values[mm_]), np.ones(mm_.sum())], sh_[mm_], rcond=None)[0][0]))
```