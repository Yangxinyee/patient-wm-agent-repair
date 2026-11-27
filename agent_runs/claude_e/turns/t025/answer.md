After removing the protocol feedback, the direct response of finger glucose to 1 u/h of insulin given in one hour is:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| change (mg/dL) | −1.99 | −2.37 | −2.01 | −1.65 | −1.12 | −0.89 | −0.61 | −0.47 | −0.26 | −0.11 |

It sums to −11.5 mg/dL; the lab-based version sums to −12.6.

Applied to the recorded doses of the 100 tuning stays at the reached rows:

| edit | estimated truth (mg/dL) | original model (mg/dL) |
|---|---|---|
| insulin x2 | −13 to −14 | −1.6 |
| insulin x1/2 | about +6.7 to +7.1 | +0.7 |

So the model understates insulin's effect roughly eightfold. The precise tail of the response matters, and I'll refine it later.

The bigger question is how to change the model's dose response by editing weights without hurting factual fit. To reason about that, and to test edits without spending forward passes, I want to rebuild the forward pass locally and check it against the tool's outputs. Step one is to read the full tensor inventory (names, shapes, any notes in tensors.json).

```python
TJ = json.load(open('/model/tensors.json'))
print(type(TJ), (list(TJ.keys())[:10] if isinstance(TJ,dict) else TJ[:3]))
for n in NAMES: print(n, SHAPES[n])
```