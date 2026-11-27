The rate follows a sliding scale with a multiplicative scatter:
- 150–180 mg/dL → base 1 U/h;
- 180–210 → base 2 U/h;
- then roughly +1 U/h per further 20–30 mg/dL, up to about 6 U/h above 310;
- around the step, the log rate scatters with sd ≈ 0.31–0.36, before rounding to 0.5 U/h.

Next I will find the exact step points with finer glucose bins and compute each record's residual against its base step. Then I will test where the scatter lives:
- the share of residual variance that is between stays versus within stays;
- the hour-to-hour correlation of residuals within a stay.

Mostly independent hourly scatter would make it a clean randomised-like dose variation. A large persistent stay component needs more care before I can trust it.

```python
mi['G'] = mi.Gtrig
fine = mi.groupby(pd.cut(mi.G, np.arange(150,420,5))).insulin.median()
print(fine.to_string())
```