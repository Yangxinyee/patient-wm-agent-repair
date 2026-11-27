The stored slots use the 4×192 layout and differ from the replica's only moderately (0.5–1.0 max abs). That can't explain 49 mg/dL errors confined to early hours.

So the likely culprit is what run_cf returns for the bundles before its restart index. Perhaps it doesn't fill them with the stored states I assume (zeros, or rows from the wrong place). I'll compare run_cf's outputs with restart at hour 12 against restart at bundle 1, bundle by bundle, for O and H.
- If O or H before index 26 differ grossly between the two calls, I'll keep only the recomputed bundles from each call: from-bundle-1 for factual rows, and from-h0 rows only for counterfactual rows.
- That fixes the dataset without depending on stored states.

```python
s=2020000024; O26,H26,_=run_cf(s,{},720,Xs); O1,H1,_=run_cf(s,{},BUN[s][1][0],Xs)
do=np.abs((O26-O1).numpy()).max(1); dh=np.abs((H26-H1).numpy()).max((1,2))
print('O diff b1-25 med %.3f max %.3f | b26+ med %.3f'%(np.median(do[1:26]),do[1:26].max(),np.median(do[26:])))
print('H diff b1-25 med %.3f max %.3f | b26+ med %.3f'%(np.median(dh[1:26]),dh[1:26].max(),np.median(dh[26:])))
print('H26 early norms', np.round(np.linalg.norm(H26[1:4].numpy().reshape(3,-1),axis=1),1), 'H1 early norms', np.round(np.linalg.norm(H1[1:4].numpy().reshape(3,-1),axis=1),1))
```