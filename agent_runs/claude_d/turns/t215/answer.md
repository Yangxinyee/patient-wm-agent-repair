On the GPU, E2 raises the factual NLL by +0.0099 nats per reading (1.5119 → 1.5218). That is about twice the 0.005 allowance.
- Every event gets worse: map_cuff +0.014, map_arterial +0.012, glucose_finger +0.007, glucose_lab +0.003.
- The MAP channel is therefore doing a lot of damage.
- Glucose mean absolute error barely moves (11.13 → 11.15), so teacher-forced glucose predictions carry no large bias.

That points to the +17 mg/dL drift appearing only in rollouts. To confirm, I will compare edited and original teacher-forced predicted means on glucose_finger readings at hours 13–15, split by whether insulin ran in the previous hour. Those are the readings where the rollout showed the bias. I will also break down where the NLL increase sits, split by event and by whether the drug ran in the previous hour.

If the hour-13 teacher-forced bias is near zero, something in the rollout itself, such as sampled values or the way a hidden reading is queried, triggers the channel. If it is large, the head generalises badly to new stays.

```python
fo9 = pd.read_csv('/work/fwd_9/factual_original.csv'); fe9 = pd.read_csv('/work/fwd_9/factual_edited.csv')
print(list(fo9.columns), fo9.shape, fe9.shape)
m9 = fo9.copy()
for c_ in fe9.columns:
    if c_ not in ('stay_id','hour','event','minute','observed','value'):
        m9[c_+'_e'] = fe9[c_].values
print(m9.head(2).to_string()[:600])
```