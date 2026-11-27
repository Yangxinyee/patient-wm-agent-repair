Ridge-fitted gates behave well for insulin:

- target rows (glucose readings with insulin in the previous hour): minimum 0.21, 1st percentile 0.79;
- all other rows: maximum 0.06, 99th percentile -0.83.

So a threshold near 0.135 with M = 100 separates them cleanly. The NE presence gate is less clean: a few rows sit on the wrong side (minimum -0.72, maximum 0.95), though the 1st and 99th percentiles are fine.

Restricting decoders to present-drug rows helps a lot:

| target | R² | slope |
|---|---|---|
| insulin surprise, lag 1 | 0.59 | 0.57 |
| insulin surprise, lag 2 | 0.24 | 0.24 |
| NE deviation, lag 1 | 0.19 | 0.25 |

At these values the full insulin lag-1 correction should improve the factual fit. The lag-2 term should cost only about +0.001 overall, and NE at 60 to 70% of full response is roughly free to +0.001.

Next I re-estimate the deficits on the present rows themselves (glucose with recent insulin, MAP with recent NE), since the scaling should use those. Then I build the gated edit and evaluate it offline.

```python
def regm(mask, cols):
    return reg(mask, cols)
print('I present', regm(isg & ins1, ['ei1','ei2','ei3']))
print('N present', regm(ism & ne1p, ['D_ne_1','D_ne_2','D_ne_3']), 'N present D1 only', regm(ism & ne1p, ['D_ne_1']))
print('var on present: ei1', round(float(RD2.loc[isg&ins1,'ei1'].var()),3), 'ei2', round(float(RD2.loc[isg&ins1,'ei2'].var()),3), 'Dne1', round(float(RD2.loc[ism&ne1p,'D_ne_1'].var()),4))
```