Scaling head 2 ×4 weakened the insulin response (−1.48 / +0.54) and made both NE readouts more negative (−0.77 / −0.71), so head 2 is not an insulin-dose head either. So far only head 0 strengthens the insulin response (−2.53 / +1.67). I'll test head 3 the same way to finish the head survey before tuning head 0's scale and checking factual cost.

```python
clear_ops(); scale_head(3, 4.0); print(len(ops()))
```