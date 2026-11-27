The measurement side is now essentially exact: median error 2.6e-7 over 493 bundles. A few bundles still miss badly (max 0.35). Before moving to actions I want to know which ones. Glucose_lab may use a different unit row from glucose_finger, or a bin tie at an edge value may be breaking.

I'll list the failing bundles with their codes and values. If they all contain glucose_lab, I'll test GL's unit index separately.

```python
bad = [(i_, e_) for i_, e_ in enumerate(errs) if e_ > 1e-4]
print(len(bad))
for i_, e_ in bad[:15]:
    print(round(e_,4), tests[i_][3], tests[i_][2])
```