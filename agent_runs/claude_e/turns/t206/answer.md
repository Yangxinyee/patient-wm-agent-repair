The readout itself generalises: on sel1 glucose_finger rows after insulin, its correlation with the true lag-1 dose deviation is −0.72 (R² ≈ 0.52, the in-sample value) and its gain is 1.18. The problem is that the factual data support only part of the gain:

| sel1 rows after insulin | Residual slope on true deviation | Factual-optimal scale of the current shift | Implied a |
|---|---|---|---|
| glucose_finger | −1.04 | 0.39 | ≈ −0.74 |
| lab glucose | −1.76 | 0.81 | ≈ −1.55 |

The counterfactual needs a ≈ −2.7. Readout noise is what makes the extra gain cost factual NLL. On the random-stay mix, the insulin part alone would cost about 0.0021 at a = −1.9, 0.0036 at −2.4 and 0.0050 at −2.8, so the NE part must probably go. Getting more response per unit of cost means cutting readout noise.

One possible source of noise is the random z draws inside the head. I'll split the variance of the insulin pair output (units 26 and 41) on those rows into a within-row part across the 16 draws and a between-row part. A large within-row share would mean a readout insensitive to the z draws could cut the cost a lot. A small share would mean the noise is in the state itself.

```python
Hc = o_c[3]  # (16, n, 192) hidden per draw, edit ins-1.9
idx6 = np.where(m1)[0][((R1.code==6) & R1.drug).values]
pr_ = (Hc[:, idx6, 26] - Hc[:, idx6, 41]).double().numpy() if isinstance(Hc, torch.Tensor) else (Hc[:, idx6, 26] - Hc[:, idx6, 41])
mean_r = pr_.mean(0); within = pr_.var(0).mean(); between = mean_r.var()
print('pair output 26-41 on 6D rows: between-row var of draw-mean %.4f, mean within-row var across draws %.4f, ratio within/total %.3f' % (between, within, within/(within+between)))
s_ = R1[(R1.code==6) & R1.drug]
print('corr(draw-mean pair, dev1t) %.3f; corr(single draw pair, dev1t) %.3f' % (np.corrcoef(mean_r, s_.dev1t)[0,1], np.corrcoef(pr_[0], s_.dev1t)[0,1]))
print('corr(draw-mean pair, dpm) %.3f' % np.corrcoef(mean_r, s_.dpm)[0,1])
```