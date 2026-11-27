The numeric inputs are `[z_value, has_value, z_dose, has_dose, z_rate, has_rate]`. Two consequences follow from the vocab:

- **Rate and dose are redundant for both drugs.** NE dose is always 4.8 × rate, and insulin dose equals rate. So z_dose = z_rate, and the dose bin equals the rate bin. The two dose/rate pathways act as one, and neither can separate insulin from NE.
- **Row 16 is the only out-of-range lever.** It is shared by both drugs, and its expected gain looks modest (maybe +0.4 to +0.8 on NE x2).

The heads mention a pending input `p` behind a gate (`pending_mlp`, `filter_gate`). If p holds the currently running infusion rates, it may be a dose-specific pathway I have not examined. Next I will grep the architecture for where p is defined and which heads read it.

```python
for l in A.split('\n'):
    if 'pending' in l or ' p ' in l or 'p)' in l or 'register' in l: print(l[:170])
```