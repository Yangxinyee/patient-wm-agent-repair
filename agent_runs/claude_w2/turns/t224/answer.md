The exact oracle of the factual-optimal correction reproduces the true dose effects closely. So the plan is validated.

| Intervention | Rollout with correction | Record-based truth |
|---|---|---|
| Insulin x2 | −16.97 | −16.51 |
| Insulin x1/2 | +8.58 | +8.26 |
| Insulin x3 | −32.0 | −33.0 |
| Insulin x1.5 | −8.74 | −8.26 |
| Insulin x0 | +14.7 | +16.5 |
| Insulin x2 from hour 16 | −12.5 | −14.2 |
| Insulin x2 from hour 6 | −24.4 | −18.6 |
| NE x2 | +4.78 | +5.61 |
| NE x1/2 | −2.76 | −2.81 |
| NE x0 | −4.16 | −5.61 |
| NE x2 from hour 6 | +5.98 | +5.37 |

The remaining errors come from shape at long lags (beyond 9 h): the model keeps growing an injected shift instead of saturating (lag-12 profile −29.7 vs −23.1). That only matters much for the hour-6 start.

So I will implement exactly this correction in the weights:
1. **Glucose channel:** gated on finger/lab glucose. Its input is a ridge decoder from the value head's input vector with_code to the combined target Σa_k·d_{t−k} + Σc_k·p(g_{t−k}).
2. **Training data for the decoders:** teacher-forced passes with random per-infusion dose multipliers. That way the decoder responds causally to the dose, not to glucose-level proxies.
3. **MAP channel:** gated on arterial/cuff MAP, built the same way.
4. **Placement:** two repurposed low-importance hidden units of `attr_heads.value_bin`. Each outputs a logit tilt along the bin centres. The gate's constant is cancelled in `embed_transition.value_bin`.

First I need the exact head code (how z, with_code and the transition table enter) and the tensor shapes. These determine where a gate vector can live and which units to repurpose.

```python
print(open('/work/head.py').read())
for k in NAMES:
    if any(s in k for s in ['attr_heads.value_bin','embed_transition','query_norm','type_emb','z_head','z_prior','embed.code','last_','heads.']):
        print(k,SHAPES[k])
```