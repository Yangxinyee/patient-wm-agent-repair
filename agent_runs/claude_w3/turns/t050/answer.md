The embedding uses several small categorical tables: action, discharged_to, rate_unit, route, status, unit, text, kq, kind and ekind. The README does not document how the atoms map onto them. A full reimplementation would mean reverse-engineering all of that, which is risky and expensive. I'll hold off unless it proves necessary.

Before choosing a mechanism, I want to check one premise. Does the original model, under teacher forcing, ignore the as-good-as-random jitter in insulin doses? I estimated the true lag-1 effect at about −2.1 mg/dL per unit/h.
- If the model's factual predictions respond to jitter by roughly that slope, it learned the drug effect, and the weak rollout response must come from somewhere else.
- If the response is near zero, the model attributes the post-insulin fall to the glucose level instead, and the edit must add dose sensitivity relative to the policy dose.

I'll compare the predicted mean against the actual next finger reading. Both are regressed on the previous reading, the band base dose, and the jitter deviation of the insulin rate.

```python
f2 = pd.read_csv('/work/fwd_2/factual_original.csv')
print(f2.shape, list(f2.columns)); print(f2.head(3).to_string())
```