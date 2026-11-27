## Step 8: a clear sliding-scale insulin protocol

**What the last result showed.** Insulin follows a step function of the last finger glucose:
- off below about 150;
- about 1 unit/h at 150–180;
- about 2 at 180–220;
- about 3 at 220–260;
- about 4 above 260.

The on/off switch is almost perfectly sharp at 150: 7.5% on in 140–150 versus 100% in 150–160.

Norepinephrine looks different. The chance of being on falls smoothly as MAP rises, with a sharper change near arterial MAP 64–65. That fits a mix of a start threshold near 65 and persistence or titration once running.

**What this means.**
- The insulin thresholds give a regression-discontinuity design. The patient's true glucose is continuous across 150, 180, 220 and 260, but the dose jumps. The jump in later glucose divided by the jump in dose identifies the causal effect, even if clinicians also respond to hidden state elsewhere.
- Rates inside a band still vary (0.5, 1.0, 1.5 …). That within-band variation might be random noise, which would be a second, more powerful source of identification. I need to check that.

**What I will do next.**
1. Examine the rate distribution inside each band.
2. Examine the fine structure around 150 (is the threshold exact?).
3. Check whether the within-band rate relates to other observables: the previous rate, lab glucose, the age of the reading, the hour.

**What would count as evidence.** If the within-band rate is independent of everything observable, it looks like pure noise. If it tracks the previous rate or other readings, it is part of the policy and may also be confounded.

```python
xi = x[x.last_glucose_finger.notna()].copy()
xi['band'] = pd.cut(xi.last_glucose_finger,[0,150,180,220,260,1000],labels=['<=150','150-180','180-220','220-260','>260'], right=True)
print(pd.crosstab(xi.band, xi.ins_rate).iloc[:, :16].to_string())
# fine look near 150 and 180 thresholds
for thr in [150,180,220,260]:
    sub = xi[xi.last_glucose_finger.between(thr-4,thr+4)]
    print(thr, sub.groupby('last_glucose_finger').agg(n=('ins_rate','size'), on=('ins_rate',lambda s:(s>0).mean()), mean=('ins_rate','mean')).T.round(3).to_string())
```