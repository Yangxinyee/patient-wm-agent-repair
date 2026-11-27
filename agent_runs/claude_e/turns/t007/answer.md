Insulin follows a sliding scale on the last finger-stick. None is given below about 140 and it is always given above 160. The mean rate steps at roughly 160, 180, 220 and 250 (about 1, 2, 2, 3, 3.8 u/h), with spread of 0.4–1.0 u/h around each step. Once started, insulin usually continues (92%).

This structure is promising for identification. If the dose is a step function of a recorded reading plus noise unrelated to the patient, I have two sources of exogenous variation: the jumps at thresholds (regression discontinuity) and the within-step noise.

Next I'll map the protocol finely in 2–5 mg/dL bins. I'll check whether it uses the current reading or the last one, whether glucose_lab or the previous rate enter, and where exactly the thresholds are.

```python
Pn = P[P.gf_now.notna()].copy()   # a finger reading at this hour, insulin decided right after
b = pd.cut(Pn.gf_now, np.arange(130,262,4))
t = Pn.groupby(b, observed=True).agg(n=('ins','size'), p_ins=('on_ins','mean'), mean_rate=('ins','mean'), sd=('ins','std'))
print(t.to_string())
# rows without a current finger reading
Po = P[P.gf_now.isna() & P.gf_last.notna()]
print('no current reading: p_ins by gf_last bin'); print(Po.groupby(pd.cut(Po.gf_last,[0,140,150,160,170,180,200,250,500]),observed=True).agg(n=('ins','size'),p=('on_ins','mean'),r=('ins','mean')).to_string())
```