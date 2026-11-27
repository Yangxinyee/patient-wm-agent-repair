This result changes the diagnosis. E2 badly corrupts the rollouts even with the recorded doses:
- **Insulin.** The edited model's factual arm runs about +18 mg/dL above the original model's at hour 13 and climbs to about +56 by hour 24. The real readings stay within about 4 of the original model's factual arm.
- **Norepinephrine.** The NE channel adds a smaller upward drift, about +1 rising to +3 mmHg.

So the dose-edit increments from fwd 8 are measured around a factual path that has run away. They are not trustworthy evidence about the channel's dose response.

A bias this size would be ruinous for the teacher-forced NLL, yet the offline estimate was only +0.006. That leaves two explanations:
1. **Rollout-specific feedback.** The model's own sampled values drive the decoded dose signal. For example, the surprise is centred on a glucose-implied dose, so in a rollout, higher sampled glucose would raise the correction, which raises glucose further.
2. **GPU and offline head disagree for the edited weights.** The edited head behaves differently on the GPU than in my offline reproduction.

To tell these apart I will look at how the hour-13 bias is spread across stays:
- A broad shift in nearly every stay, already at the first hidden reading, would point to explanation 2 or to a gate leak.
- A few runaway stays would point to sampling blow-up.

```python
fe_ = d8e[d8e.edit=='insulin_x2'][['stay_id','hour','factual']].copy()
fo_ = d8o[d8o.edit=='insulin_x2'][['stay_id','hour','factual']].copy()
fe_['bias'] = fe_.factual.values - fo_.factual.values
for h_ in [13, 14, 16, 20, 24]:
    b_ = fe_[fe_.hour==h_].bias.values
    print(h_, 'n', len(b_), 'pct5/25/50/75/95', np.round(np.percentile(b_, [5,25,50,75,95]),1), 'frac>10', np.round((b_>10).mean(),2))
b13 = fe_[fe_.hour==13].set_index('stay_id').bias
ins12_ = pd.Series({s_: INS_U[sid2i[s_], 12] for s_ in b13.index}); ins11_ = pd.Series({s_: INS_U[sid2i[s_], 11] for s_ in b13.index})
g12_ = pd.Series({s_: GF_U[sid2i[s_], 12] for s_ in b13.index})
print('bias13 by ins12>0:', b13[ins12_>0].mean().round(1), b13[ins12_==0].mean().round(1), (ins12_>0).sum())
print('bias13 by ins11>0:', b13[ins11_>0].mean().round(1), b13[ins11_==0].mean().round(1))
print('corr bias13 with g12', np.round(np.corrcoef(b13.values[~np.isnan(g12_.values)], g12_.values[~np.isnan(g12_.values)])[0,1],2))
print('orig fact h13 mean', fo_[fo_.hour==13].factual.mean().round(1), 'edit', fe_[fe_.hour==13].factual.mean().round(1))
```