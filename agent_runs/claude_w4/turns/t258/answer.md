Two useful things from this.

First, the true impulse responses I estimated earlier, per unit rate:
- **Insulin → finger glucose:** about −1.8, −2.1, −1.9, −1.6, −1.1, −0.8 mg/dL at lags 1–6, fading to zero by about lag 9. A sustained doubling therefore reaches a plateau of roughly −10 mg/dL per unit rate after 6–8 hours.
- **NE → arterial MAP:** +3.7, +2.7, +1.3, +1.1 mmHg… a faster response, with a plateau of about +10 per unit rate.

This means the model's injected shift must produce an effect of roughly this size and shape in the rollout.

Second, my hour profile mixed all four edits together, so it can't tell me anything yet. I'll split it per edit and compare edited against original hour by hour. This answers the open question: does the v3 drug‑bundle injection (about −1 mg/dL at the next reading, by the ridge estimate) show up at hour 13 of the rollout, and does it persist?
- If the edited‑minus‑original insulin ×2 shift is about −1 at hour 13 and then fades, the injection works but the rollout washes it out.
- If it is about 0 even at hour 13, the ridge estimate overstated the immediate effect, and the carrier is too weak.

I'll also compute the factual cost of a constant‑centred injection, using a Gaussian approximation on the fwd_2 residuals. That tells me whether state‑dependent centring is needed.

```python
D6={m:pd.read_csv(f'/work/fwd_6/dose_edit_{m}.csv') for m in ['original','edited']}
for e in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    a_=D6['original']; b_=D6['edited']
    pa=a_[(a_.edit==e)&a_.reached].groupby('hour')['shift'].mean(); pb=b_[(b_.edit==e)&b_.reached].groupby('hour')['shift'].mean()
    print(e,'orig',pa.round(2).values,'\n   edit',pb.round(2).values)
# factual: insulin rate at each insulin bundle (tuning stays), and finger residuals after insulin bundles
tuall=pd.read_csv('/data/tuning.csv')
ins=tuall[tuall.event=='insulin'][['stay_id','hour','rate']]
print('insulin rows',len(ins),'rate mean %.2f sd %.2f'%(ins.rate.mean(),ins.rate.std()), np.round(np.quantile(ins.rate,[.05,.25,.5,.75,.95]),2))
```