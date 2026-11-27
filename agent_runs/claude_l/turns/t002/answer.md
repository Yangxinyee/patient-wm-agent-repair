I now know how the model works. Drug atoms are encoded through `encode_action.*` into u_k. u_k enters the latent posterior `z_post([ctx, e_k, u_k])` and the prelude token list. Because the rollout changes only the insulin and norepinephrine atoms' rate and dose (their z-scores and bins), every effect of a dose edit has to flow through the action embedding: `embed.numeric` columns for z_dose and z_rate, `embed.bins.dose` and `embed.bins.rate` for the drug codes, then `encode_action`, and from there into z_post and the prelude.

Next I'll look at vocab.json (codes, normalisation, bin edges) and the tensor list, so I know which parameters sit on that path. Then I'll move to the data.

```python
V = json.load(open('/model/vocab.json'))
print(V.keys())
for k,v in V.items():
    s = json.dumps(v)
    print(k, ':', s[:1500])
    print('---')
T = json.load(open('/model/tensors.json'))
print(type(T), len(T))
print(json.dumps(T)[:3000])
```