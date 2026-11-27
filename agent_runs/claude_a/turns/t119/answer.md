The capture itself completed. The error came from my assembly step: at later bundle positions the batched call includes only the stays that still have bundles left, so the batch index is not the stay index.

**Next step:** redo only the assembly, mapping each stay to its position among the stays still active at that bundle position. The captured data is already in `CAP`. Then check alignment against `tf_states_ds`'s output and print the atom and bundle columns.

**What success looks like:** batch sizes equal the active-stay counts, and the alignment error is ≈ 0.

```python
lens = np.array([len(r) for r in dsA['srows']])
ok_sizes = all(CAP[k][0].shape[0] == int((lens > k).sum()) for k in range(len(CAP)))
print("batch sizes match active counts:", ok_sizes, " ncalls", len(CAP), " max len", lens.max())
EA = torch.zeros(nbA, 4, 192); YA = torch.zeros(nbA, 4, 192)
for k in range(len(CAP)):
    act_ = np.where(lens > k)[0]
    for pos, s_i in enumerate(act_):
        rw = dsA['srows'][s_i][k]
        EA[rw] = CAP[k][2]['E'][pos]; YA[rw] = CAP[k][0][pos]
print("align max|dY|", float((YA-HnA).abs().max()), " time %.1fs"%(time.time()-t0))
print(dsA['at'].columns.tolist()); print(dsA['at'].head(8).to_string())
print(dsA['bd'].columns.tolist()); print(dsA['bd'].iloc[3:9].to_string())
```