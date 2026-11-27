Numeric ×5 costs +0.0016 nats per reading, about a third of the budget. glucose_lab (+0.005) and map_arterial (+0.003) are hurt more than finger glucose (+0.002). The input-side lever is therefore not free. Before stacking a head-0 gain on it, I want to know whether its factual changes go in the causally right direction. The check: the lag-1 slope of the prediction change at finger-glucose readings on the preceding dose's within-tier deviation. That slope should be negative; the data say about -2 per U/h. I'll inspect the existing per-reading table `gu` that I built earlier for this.

```python
print(list(gu.columns)); print(gu.head(3).to_string()[:900])
```