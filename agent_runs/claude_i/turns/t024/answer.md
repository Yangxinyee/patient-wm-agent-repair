Norepinephrine raises MAP. Per 1 mcg/kg/min, the rate coefficient is +3.2 at lag 1, +5.0 at lag 2 and +5.6 at lag 3, with arterial and cuff readings agreeing.

These are total effects. Titration is multiplicative on the previous rate, so a high rate now carries forward into later rates. The direct kernel therefore needs the same feedback deconvolution as insulin, but here the dose persistence D_m will be large.

I'll compute, over 12 lags, the effect of the random current rate on later arterial and cuff MAP (T_k) and on later rates (D_m), then solve for the direct kernel K_k. Arterial MAP is recorded nearly every hour in arterial-line stays, so it gives an almost selection-free outcome.

```python
Nn = P.copy()
gN = Nn.groupby(level='stay_id')
L=13
for k in range(1,L):
    Nn[f'ma_l{k}'] = gN.ma.shift(-k); Nn[f'mc_l{k}'] = gN.mc.shift(-k); Nn[f'n_l{k}'] = gN.ne0.shift(-k)
Nn['mdec'] = Nn.ma.combine_first(Nn.mc)
Nn['m_prev'] = gN.m_last.shift(1)
Nn['ne_prev2'] = gN.ne0.shift(2)
Nn = Nn[Nn.index.get_level_values('h')>=1]
SN = Nn[(Nn.ne_prev>0) & Nn['ne'].notna() & Nn.mdec.notna() & Nn.m_prev.notna()].copy()
SN['b1'] = (SN.mdec<65).astype(float); SN['b3'] = (SN.mdec>=75).astype(float); SN['lprev']=np.log(SN.ne_prev)
def effN(s, outc):
    X = [s.mdec, s.m_prev, s.b1, s.b3, s.ne_prev, s.lprev, s.ne_prev2, s['ne']]
    res, n, sd = ols(s[outc], X, ['mdec','mprev','b1','b3','neprev','lprev','neprev2','rate'])
    return res.loc['rate','b'], res.loc['rate','se'], n
TNa={}; TNc={}; DN={0:1.0}
for k in range(1,L):
    TNa[k]=effN(SN,f'ma_l{k}'); TNc[k]=effN(SN,f'mc_l{k}'); DN[k]=effN(SN,f'n_l{k}')[0]
KNa = deconv({k:TNa[k][0] for k in TNa}, DN, L); KNc = deconv({k:TNc[k][0] for k in TNc}, DN, L)
print(" k   T_ma(se)       T_mc(se)       D_k     K_ma    K_mc")
for k in range(1,L):
    print(f"{k:2d}  {TNa[k][0]:6.2f}({TNa[k][1]:.2f})  {TNc[k][0]:6.2f}({TNc[k][1]:.2f})  {DN[k]:6.3f}  {KNa[k]:6.2f}  {KNc[k]:6.2f}")
```