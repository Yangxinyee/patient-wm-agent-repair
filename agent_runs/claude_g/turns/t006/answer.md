The insulin policy is a sliding scale on the glucose_finger reading taken 5 minutes earlier:
- below about 140 there is no insulin;
- from 140 to 160, insulin is given 57% of the time at about 1 u/h;
- from 160 to 200, every reading gets insulin, at about 1 then 2 u/h; above 200 the rate rises in steps (2, 3, 4, 6 u/h).

The rate varies a lot within each glucose band (SD about a third of the mean). That spread is the variation I need to identify the effect, provided it is random rather than tied to an unrecorded patient trait such as insulin resistance, which would also drive glucose.

Insulin is almost always given right after a glucose_finger reading. Norepinephrine is commoner with an arterial line: 8,068 stays have both, 5,158 have norepinephrine with cuff MAP only.

Next I will map the insulin scale finely, at 5 mg/dL resolution and against the previous hour's rate, to find its exact form and what explains the residual. I will also check the norepinephrine policy against MAP. The key test: if the residual (rate ÷ scale) is strongly autocorrelated within a stay, it reflects a patient trait (possible confounder); if it is independent from hour to hour, it is fresh randomisation.

```python
g['g5'] = (g.glucose_finger//10*10)
t = g.groupby('g5').agg(n=('ins','size'), p_ins=('insulin', lambda x: x.notna().mean()), mean_rate=('insulin','mean'), sd=('insulin','std'), mn=('insulin','min'), mx=('insulin','max'))
print(t[(t.index>=130)&(t.index<=350)].round(3).to_string())
# Lagged insulin rate as predictor
Q['ins_prev'] = Q.groupby(level='stay_id').ins.shift(1)
Q['gf_prev'] = Q.groupby(level='stay_id').glucose_finger.shift(1)
gg = Q[Q.insulin.notna() & Q.glucose_finger.notna()].copy()
print(np.corrcoef(gg.ins, gg.glucose_finger)[0,1], np.corrcoef(gg.ins, gg.ins_prev.fillna(0))[0,1])
```