Splitting the insulin x2 shift by each stay's recorded rate shows N5 does not scale up uniformly. It strengthens low-dose stays (band ≤1.5 U/h: -0.82 → -2.19). It weakens high-dose stays (2.5-3.5 U/h: -2.70 → -0.98), and N20 does the same. So x5 pushes doubled high doses into a saturated, partly reversed region, while the true effect grows linearly with dose. This points to a milder factor, around x2 or x3, as possibly better for the main readout and cheaper. Next I'll get the record-based truth per band, then the insulin-half and NE breakdowns.

```python
print(inspect.signature(true_in), inspect.signature(true_ne), inspect.signature(pop_true))
```