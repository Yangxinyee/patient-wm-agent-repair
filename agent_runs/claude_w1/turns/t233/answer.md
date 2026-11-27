q_of returns only the query q, so the train rows Xf are q vectors. The real head input adds the code embedding, last-value bin embedding, last_status and last_numeric (built from last-reading age and last value). The first three are constant within a (code, last bin) cell, so the per-cell compensation removes them; that is the -215 offset. last_numeric varies within a cell, which explains the inflated spread on setA.

Fix: make the decoders orthogonal to the last_numeric output subspace. If last_numeric is linear in its 8 inputs, that subspace has dimension 8 or less and the removal is exact. I will check the rank of last_numeric outputs on setA (lnum) and how much of each decoder's setA spread comes from it.

```python
U_,S_,Vt=np.linalg.svd(lnum-lnum.mean(0),full_matrices=False); print('lnum sv',np.round(S_[:10],2))
k_=int((S_>1e-6*S_[0]).sum()); print('rank',k_)
B_=Vt[:min(k_,12)]
for nm,w0 in [('ins',wI),('ne',wN)]:
    print(nm,'|w|',round(np.linalg.norm(w0),1),'sd w.lnum',round(np.std(lnum@w0),2),'sd w.(wc-lnum) by code', [round(np.std((wc_-lnum)[codeA==c]@w0),2) for c in range(4)])
```