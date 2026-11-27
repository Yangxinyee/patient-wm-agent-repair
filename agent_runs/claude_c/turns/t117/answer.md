The cap breakdown changes the picture. Of the 42 tuning stays with norepinephrine from hour 12 on, 19 are capped at 1.0.

| Group | Original x2 | Original x1/2 | Dose columns x5: x2 | Dose columns x5: x1/2 |
|---|---|---|---|---|
| Norepinephrine, capped | −0.70 | −1.07 | ≈0 | ≈0 |
| Norepinephrine, not capped | −0.20 | +0.11 | ≈0 | ≈0 |
| Insulin, high dose | −2.42 | +1.35 | −2.54 | +2.60 |
| Insulin, low dose | −0.88 | −0.19 | −2.46 | +0.06 |

- **Norepinephrine:** non-capped stays are also wrong-signed, so there is no right-signed but weak pathway to amplify.
- **Insulin:** x2 is stuck near −2.5 whatever the input gain, and the low-dose response rises to that same level. That looks like a ceiling on how much the action token's dose rotation can move glucose, against a true value of about −13.6.

Pass 10 left the factual results unchanged to three decimals. I want to know whether the edited predictions differ from the original by exactly zero or by a measurable amount that tracks the previous surprise:
- If exactly zero, the slots do not attend to the z token at all, and the z route is closed for good.
- If small but systematic, the z token's content is just weak relative to its bias, and a much larger amplification could still make it a selective channel.

This needs no forward pass, only the per-reading predictions from pass 10.

```python
fo = pd.read_csv('/work/fwd_10/factual_original.csv'); fe = pd.read_csv('/work/fwd_10/factual_edited.csv')
dpm = fe['predicted_mean'].to_numpy() - fo['predicted_mean'].to_numpy()
print('pred diff: max abs %.5f, rms %.5f' % (np.abs(dpm).max(), np.sqrt((dpm**2).mean())))
fo['dpm']=dpm; fo['surp']=fo['observed']-fo['predicted_mean']
for ev in ['glucose_finger','map_arterial']:
    x = fo[fo.event==ev].sort_values(['stay_id','hour']).copy()
    x['prev_surp'] = x.groupby('stay_id')['surp'].shift(1)
    x = x.dropna(subset=['prev_surp'])
    b = np.polyfit(x['prev_surp'], x['dpm'], 1)[0]
    print(ev, 'slope of pred change on previous surprise %.5f ; rms dpm %.5f' % (b, np.sqrt((x.dpm**2).mean())))
```