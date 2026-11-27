The z posterior on the failing bundles still matches closely (median error 3e-4). So my e and u for these bundles are at most slightly off, yet the next h is clearly wrong.

The failures all share one feature: atoms of both drug codes in the same bundle. At hour 25.17 that is two END records; at the h+5min bundles it is two STARTs and/or two ENDs. With a single atom, a sum, a mean and a sum divided by the square root of n give the same result. So my earlier checks could not tell how a set of two drug atoms is combined. The likely culprit is the set reduction or the END routing when there are two such atoms.

Next I will rebuild the END-atom vectors and the set encoder directly from the weights. I will first check them against my stored e on a bundle with a single END record. Then, on the two-END bundles at hour 25.17, I will try variants: sum divided by the square root of n, mean, sum, and each END atom alone. The variant that brings the one-step h error to ~0 settles the rule.

```python
k_ex = list(atoms_by_bundle.keys())[:2]
print('atoms_by_bundle key example', k_ex, '->', atoms_by_bundle[k_ex[0]][:4] if len(atoms_by_bundle[k_ex[0]]) else None)
print('n keys', len(atoms_by_bundle))
print('encode tensors', [ (n_, SHAPES[n_]) for n_ in NAMES if n_.startswith('encode.')])
```