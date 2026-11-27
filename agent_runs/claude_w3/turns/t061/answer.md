With the increment measured from the last glucose reading of either kind, the costs are:
- full dose: 0.0046 nats, borderline;
- jitter deviation: −0.0010, an improvement;
- dose minus a constant 1.5 u/h, applied only while insulin runs: −0.0002, neutral.

So the edit does not need the band. A signal c·(dose − 1.5) computed from the insulin atom alone, applied as an increment, keeps the factual fit.

How that increment adds up in the rollout depends on how strongly the model anchors on its own sampled readings. If the next prediction ≈ (1−K)·previous prediction + K·latest reading, only a fraction K of each increment carries forward, so the kernel must be designed for the actual K. I'll estimate K from fwd_2. For consecutive hourly finger readings, I regress the prediction at t+1 on the prediction at t and the reading at t. I'll do the same for map_arterial.
- K near 1 means simple hourly increments suffice.
- A clearly smaller K means the head shift must be the effect minus a running average of past effects.

```python
def anchor_fit(ev_):
    s_ = f2[f2.event==ev_].copy(); s_['ih']=s_.hour.round().astype(int); s_=s_.sort_values(['stay_id','ih'])
    s_['m_prev']=s_.groupby('stay_id').predicted_mean.shift(1); s_['y_prev']=s_.groupby('stay_id').observed.shift(1)
    s_['h_prev']=s_.groupby('stay_id').ih.shift(1)
    s_=s_[(s_.ih-s_.h_prev==1)&(s_.ih>=1)].dropna()
    X_=np.column_stack([s_.m_prev, s_.y_prev, np.ones(len(s_))]); b_=np.linalg.lstsq(X_, s_.predicted_mean.values, rcond=None)[0]
    return len(s_), np.round(b_,3)
for ev_ in ['glucose_finger','map_arterial','glucose_lab']:
    print(ev_, anchor_fit(ev_))
```