No bundle had an action and zero measurement atoms. Insulin bundles evidently also carry the END atom of the previous infusion in `e`. I'll print a few bundles' time, counts and labels for this stay, so I can select the insulin START bundles correctly.

```python
print([(round(b_['t'], 3), b_['nM'], b_['nA'], b_['labels']) for b_ in Bg[:14]])
```