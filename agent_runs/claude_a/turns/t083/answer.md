On the same 456/305 reached stay-hours, my current effect kernels imply these targets:
- insulin x2: −14.3
- insulin x½: +7.2
- norepinephrine x2: +4.1
- norepinephrine x½: −2.0

The original model gives −1.9 / +1.0 / −0.4 / −0.5. It understates the insulin response about 7-fold and gets the norepinephrine x2 direction wrong.

These targets assume each drug's effect is linear in its rate and does not depend on the glucose or MAP level. The x2 and x½ readouts test exactly that, so I'll check it before designing the edit. Next I'll look at my prepared stay-hour panel `H`, then measure the within-tier effect of insulin rate separately by glucose tier. If the per-unit slope is about the same across tiers (glucose ~165 to ≥250), the linear additive form holds. If it scales with the glucose level, the edit must make the shift scale with glucose too.

```python
print("H cols:", list(H.columns))
print(H.iloc[:2].T.head(60).to_string()[:3000])
```