Splitting the sel1 out-of-sample cost shows where it comes from:
- **Insulin part:** costs +0.0072 / +0.0121 / +0.0170 per glucose_finger row after insulin at a = −1.9 / −2.4 / −2.8. A quadratic fit puts the factual-optimal gain near −0.15, far from the gain the counterfactual needs (about −2.7).
- **Lab glucose after insulin:** these rows improve (−0.006 to −0.002).
- **NE part:** costs +0.018 to +0.024 per map_arterial row. That alone is +0.0061 per reading on sel1.
- **Gates:** rows without a drug are unaffected by the gain, so the gates themselves are fine.

The puzzle: my regression said the model under-uses the dose deviation by about −1.7 per u/h. A unit-gain readout carrying that signal should not cost this much on finger rows. Either the readout decodes the deviation much worse out of sample than the in-sample R² of 0.52 suggests, or the finger residuals do not carry the missing effect.

To find out, I'll compute the true lag-1 dose deviation for the sel1 readings from the records. Then I'll regress the original model's residuals on it and on the edit's mean shift, separately for finger and lab rows, and check how well the shift tracks the deviation.

```python
o_c = head_run(X, RR, gated3(-1.9, 0.0))
R1['pm_o'] = np.asarray(o_o[0], float)[m1]; R1['dpm'] = np.asarray(o_c[0], float)[m1] - R1.pm_o
R1['res'] = R1.obs.astype(float) - R1.pm_o
stu = list(Gtu['gf'].index); 
def tier5(g):
    return np.where(np.isnan(g), 5, np.where(g<150,0,np.where(g<180,1,np.where(g<220,2,np.where(g<300,3,4)))))
EtabL = np.array([0,1.03,2.07,3.24,6.24,.03])
hh = np.floor(R1.hour.values.astype(float)).astype(int) - 1
gfI = Gtu['gf']; insI = Gtu['ins']
def gget2(G, s, h):
    try: return float(G.loc[s, h])
    except Exception: return np.nan
g_prev = np.array([gget2(gfI, s, h) for s, h in zip(R1.stay.values, hh)])
d_prev = np.array([gget2(insI, s, h) for s, h in zip(R1.stay.values, hh)])
d_prev = np.nan_to_num(d_prev)
R1['dev1t'] = d_prev - EtabL[tier5(g_prev)]
for c_ in [6, 10]:
    s_ = R1[(R1.code==c_) & R1.drug]
    X_ = np.c_[np.ones(len(s_)), s_.dev1t]; b_res = np.linalg.lstsq(X_, s_.res, rcond=None)[0]
    X2_ = np.c_[np.ones(len(s_)), s_.dpm]; b_dpm = np.linalg.lstsq(X2_, s_.res, rcond=None)[0]
    gain_ = np.linalg.lstsq(X_, s_.dpm/(-1.9), rcond=None)[0]
    print(c_, 'n', len(s_), 'sd dev1t %.3f' % s_.dev1t.std(), '| res~dev1t slope %.3f' % b_res[1], '| res~dpm slope %.3f' % b_dpm[1],
          '| readout(dpm/-1.9)~dev1t gain %.3f' % gain_[1], 'corr %.3f' % np.corrcoef(s_.dpm, s_.dev1t)[0,1], '| sd res %.2f' % s_.res.std())
```