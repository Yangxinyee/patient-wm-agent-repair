On random tuning stays the persistence deficit is small. The within-tier slope is +2.2 per 50 mg/dL (se 0.78, without allowing for clustering within stays). A correction would gain about 0.0004 per reading and rests on a 2–3 se estimate. Correcting it per bin would mostly fit noise, so I'm dropping the persistence idea.

That leaves choosing the insulin gain. Projected per-reading cost on random batches and insulin ×2 as a share of truth:

| a | Projected cost | Insulin ×2 / truth |
|---|---|---|
| −2.6 | ≈ 0.0042 | ≈ 99% |
| −2.5 | ≈ 0.0039 | ≈ 95% |
| −2.4 | ≈ 0.0036 | ≈ 91% |

Failing the factual check would be far worse than a few percent on the main readout. So unless I can cut the cost per unit of gain, I should settle near −2.45 to −2.5.

One remaining lever is overfitting of the readout. It was fitted on the 500 train stays. If its in-sample correlation with the true deviation is clearly above the out-of-sample 0.72 on sel1, refitting on more stays (more states passes, of which I have plenty) would lower the noise and the cost. I'll measure the in-sample correlation and gain on the train datasets' finger rows after insulin.

```python
Eu = ins_only(-2.6)
cin = []
for i, (Xs_i, Rr_i) in enumerate(datasets):
    o_i = head_run(Xs_i, Rr_i, Eu)
    dpm_i = np.asarray(o_i[0], float) - np.asarray(base_out[i][0], float)
    msk_i = (Rr_i.code.values==6) & A_.loc[A_.index[sum(len(datasets[j][1]) for j in range(i)) + np.arange(len(Rr_i))], 'hi'].values.astype(bool)
    cin.append(pd.DataFrame({'dpm': dpm_i[msk_i], 'dev1': Rr_i.dev1.values[msk_i]}))
CI = pd.concat(cin, ignore_index=True)
print('in-sample 6D rows', len(CI), 'corr(dpm, dev1) %.3f' % np.corrcoef(CI.dpm, CI.dev1)[0,1], '| gain dpm/dev1 %.3f' % np.polyfit(CI.dev1, CI.dpm, 1)[0], '| sd dev1 %.3f' % CI.dev1.std())
s_ = R1[(R1.code==6) & R1.drug]
o_s1 = head_run(X, RR, Eu); dpm_s1 = np.asarray(o_s1[0], float)[m1][((R1.code==6) & R1.drug).values] - s_.pm_o.values
print('sel1 out-of-sample corr(dpm, dev1t) %.3f gain %.3f' % (np.corrcoef(dpm_s1, s_.dev1t)[0,1], np.polyfit(s_.dev1t, dpm_s1, 1)[0]))
```