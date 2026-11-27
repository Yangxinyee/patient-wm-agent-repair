The prepared `ne` frame only has hours with NE activity. MAP comes from either an arterial line or a cuff, so the policy and the effect act on whichever MAP is measured.

**Next step:** build a full hourly panel from the raw training records:
- M = arterial MAP if present, else cuff;
- N(h) = NE rate started at hour h, or 0 if none.

Then fit the factual best linear predictor, MAP(h+1) on MAP(h), MAP(h−1) and N(h)…N(h−5), with policy-zone dummies. I will compare its NE coefficients with the causal direct kernel implied by ψ and the fitted AR coefficients, k_l = ψ_l − Σ a_i ψ_{l−i}.

**What would confirm or refute it:**
- Factual coefficients ≈ causal: little severity confounding, so a level-based NE injection costs little.
- Large negative excess at lags ≥ 2: severity confounding, so it costs a lot.

```python
d0 = tr[tr.event.isin(['map_arterial','map_cuff'])][['stay_id','hour','event','value']].copy(); d0['h']=np.round(d0.hour).astype(int)
pv = d0.pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
nn = tr[tr.event=='norepinephrine'][['stay_id','hour','rate']].copy(); nn['h']=np.floor(nn.hour+1e-6).astype(int)
nn = nn.groupby(['stay_id','h']).rate.first()
grid2 = tr[['stay_id']].drop_duplicates().merge(pd.DataFrame({'h':np.arange(-1,25)}), how='cross').set_index(['stay_id','h'])
grid2 = grid2.join(pv).join(nn.rename('N')); grid2['N']=grid2.N.fillna(0.0)
grid2['M'] = grid2['map_arterial'].fillna(grid2['map_cuff'])
g2 = grid2.reset_index().sort_values(['stay_id','h'])
gb = g2.groupby('stay_id')
for L in range(1,7): g2[f'N{L}'] = gb.N.shift(L).fillna(0.0)
g2['M1'] = gb.M.shift(1); g2['M2']=gb.M.shift(2); g2['Mn'] = gb.M.shift(-1)
cc_ = g2.dropna(subset=['Mn','M','M1','M2']).copy()
cc_['z_lo'] = (cc_.M<65).astype(float); cc_['z_hi']=(cc_.M>=75).astype(float)
Xc = np.c_[np.ones(len(cc_)), cc_[['M','M1','M2','z_lo','z_hi','N','N1','N2','N3','N4','N5']].values]
bopt, *_ = np.linalg.lstsq(Xc, cc_.Mn.values, rcond=None)
res_ = cc_.Mn.values - Xc@bopt
print("n", len(cc_), " resid sd %.3f"%res_.std())
print("AR M,M1,M2:", np.round(bopt[1:4],3), " zones lo/hi:", np.round(bopt[4:6],3))
print("factual N coefs lag1..6:", np.round(bopt[6:],3))
a_ = bopt[1:4]; ps = np.r_[psi_ne[:6]]
kdir = [ps[l] - sum(a_[i]*ps[l-1-i] for i in range(3) if l-1-i>=0) for l in range(6)]
print("causal psi:", np.round(ps,3), "\ncausal direct kernel (AR-implied):", np.round(kdir,3))
print("N sd (all rows) %.3f, frac rows N>0: %.3f"%(cc_.N.std(), (cc_.N>0).mean()))
```