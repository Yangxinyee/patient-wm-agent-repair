Extending the insulin regression to six lags clarifies the picture. The factual deficits are:

| lag | deficit | se |
|---|---|---|
| 1 | -1.3 | 0.3 |
| 2 | -1.3 | 0.3 |
| 3 | -0.4 | 0.3 |
| 4 to 6 | +0.3 to +0.6 | about 0.3 |

At lags 4 to 6 the model slightly over-propagates. So, given real readings that carry the effect, the model already follows the later lags. What it misses is the first two hours. That supports fixing lags 1 and 2 at the head and letting rollout feedback through the sampled readings carry the rest.

The variances needed for cost estimates are 0.29 for the lag-1 insulin surprise (glucose rows) and 0.022 for the lag-1 NE deviation (MAP rows).

Before building the edit I need a faithful offline version of the tool's factual NLL. The tool draws z from the prior four times and averages probabilities, and z enters with_code through z_head. So I will rebuild with_code from (context, z) and check whether the stored WC0 used the prior mean of z. Then I can measure the extra noise any decoder would pick up from z sampling.

```python
print([n for n in NAMES if any(s in n for s in ['z_head','query_norm','type_emb','last_numeric','out_norm'])])
```