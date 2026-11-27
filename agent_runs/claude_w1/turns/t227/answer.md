The hidden activations match exactly, so W1 and b1 are right, but the transition-row match fails. Either the stored TR, W2 or b2 is stale, or lg is defined differently. I will reload the value-head tensors from the original weights and redo the check, this time comparing renormalised probabilities directly.

```python
nm_=[n for n in NAMES if 'value_bin' in n or 'transition' in n]; print(nm_)
for n in nm_: print(n, SHAPES[n])
```