The z error is about 1 at every bundle, so the earlier near-exact match must have rested on a different setup. Possibilities:
- `X` is not aligned with `BUN` the way I assume;
- the stored rows hold the state before each bundle, not after;
- run 66 compared against a different array.

Next I load the states file from forward pass 2 directly and check its keys and its row metadata (stay, hour, drug flags) against `BUN`. Then I compute z using the state from row k−1 and from row k. One of them should give errors near zero; that alignment is the one to use. If neither does, the problem is in the ctx computation, and I will look at `zerr` for the formula that worked.

```python
npz=np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
print(npz.files)
for f in npz.files: print(f, npz[f].shape, npz[f].dtype, short(npz[f][:6],300))
Xs=npz[[f for f in npz.files if npz[f].ndim==2 and npz[f].shape[1]==992][0]]
print('X same as file?', Xs.shape, np.abs(Xs[:len(X)]-X).max() if Xs.shape==X.shape else 'shape differs')
print(open('/work/fwd_2/summary.json').read()[:800])
print(inspect.signature(zerr), zerr.__code__.co_names, [c for c in zerr.__code__.co_consts if isinstance(c,(int,float,str))])
```