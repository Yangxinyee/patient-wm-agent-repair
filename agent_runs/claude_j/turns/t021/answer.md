The MAP measurement design is clean.

- **Arterial line:** about 48% of stays have one, with map_arterial in about 91% of hours whatever the MAP. The rest never have map_arterial.
- **Cuff:** map_cuff appears in about 86% of hours in stays without an arterial line, also MAP-independent, and occasionally in stays with one.
- **Stopping:** norepinephrine stops only when the rate has been weaned below about 0.1, mostly at 0.02-0.05 or below.

So MAP outcomes have no measurement selection by value, and the norepinephrine rate is a capped, titrated random walk.

Next is the norepinephrine analogue of the insulin analysis. I will form the expected rate given the previous rate, the current MAP band and which MAP was read, using cells to respect the 1.0 cap and rounding. Then I will:
1. take the level surprise u_t = rate_t − expected rate, and check that it is uncorrelated with earlier readings;
2. regress future map_arterial, future map_cuff and future rates on it, and deconvolve.

A clear positive impulse response would mean norepinephrine raises MAP. A flat zero would mean the drug does nothing and the rate-MAP association is pure confounding.

```python
z = P[(P.h>=1)].copy()
z['src'] = np.where(z.MA.notna(),'MA',np.where(z.MC.notna(),'MC','none'))
z['mapv'] = np.where(z.MA.notna(), z.MA, z.MC)
z['band'] = np.where(z.src=='none','none', np.where(z.mapv<65,'lo',np.where(z.mapv<=75,'mid','hi')))
# careful thresholds approx; exclude exact 65/75 for cleanliness later
c = z[(z.N0_prev>0)].copy()
c['Nb'] = pd.qcut(c.N_prev, 25, duplicates='drop')
cell = c.groupby(['Nb','band','src'], observed=True).N0.transform('mean')
c['uN'] = c.N0 - cell
print("uN sd:", c.uN.std(), " n:", len(c))
for col in ['GF','MA_ff','MC_ff','h','GL_ff']:
    ss=c[[col,'uN']].dropna(); print("corr uN with",col, round(ss.corr().iloc[0,1],4))
c['MA_m1']=c.groupby('stay_id').MA.shift(1)
ss=c[['MA_m1','uN']].dropna(); print("corr uN with MA_m1", round(ss.corr().iloc[0,1],4), len(ss))
both = P[P.MA.notna()&P.MC.notna()]
print("MC-MA same hour: mean", round((both.MC-both.MA).mean(),2), "sd", round((both.MC-both.MA).std(),2), "corr", round(both[['MA','MC']].corr().iloc[0,1],3))
Cn = c
```