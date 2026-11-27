Three things about the norepinephrine rule:
- **Cap:** about 14% of infusion-hours sit at the 1.0 cap.
- **Start rate:** higher at lower MAP.
- **Thresholds:** the titration has sharp edges at 65 and 75. Below 65 the rate rises, at 65–74 it is held (mean log-ratio ≈ 0), and above 75 it falls (≈ −0.25).

Within the below-65 band, the mean log-ratio grows with MAP (0.10 at 56, 0.20 at 64). That suggests additive steps in rate rather than multiplicative ones, since lower-MAP patients sit at higher rates. Stopping happens mainly when the rate is very small (≤ 0.05 u... mcg/kg/min).

The exact form matters because I need a clean "innovation" (the random part of each dose) to estimate the effect on MAP. Next I'll tabulate the rate change by MAP band (below 65, 65–75, above 75) and previous-rate bin, in both additive and log form. Whichever form makes the noise homogeneous is the right one.

```python
cont = Nall[(Nall.ne_prev_on==1)&Nall.m.notna()].copy()
cont['d'] = cont.norepinephrine - cont.ne_prev
cont['lr'] = np.log(cont.norepinephrine/cont.ne_prev)
cont['band'] = np.where(cont.m<65,'low',np.where(cont.m<75,'mid','high'))
cont['pb'] = pd.cut(cont.ne_prev,[0,0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.99,1.0])
print(cont.groupby(['band','pb']).agg(d=('d','mean'),dsd=('d','std'),lr=('lr','mean'),lrsd=('lr','std'),n=('d','size'),capfrac=('norepinephrine',lambda x:(x>=1).mean())).round(3).to_string())
```