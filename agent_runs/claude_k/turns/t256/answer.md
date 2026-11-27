The saved edit matches writer v2 from my notes: only z_post hidden units 12, 88 and 105 are changed, with a large gate weight (about 1000) and output columns that write into z. The kernel no longer holds `lamA`/`lamB`; it holds `lamA0`/`lamB0`, and `j0` has been overwritten with 3. Next I'll print small summaries only: the gains, the impulse-response arrays (`hG`, `hM`, `KAP`), and each edited unit's input segment norms and bias. I'll use these to compute the "truth" on the stays I send to dose_edit. If `hG`/`hM` hold the per-hour impulse responses behind my truth estimates (-13.8 / +4.6), I can reuse them directly.

```python
for k in ['lamA0','lamB0','hG','hM','KAP','sG','sM','coefz','dB','mlt','time']:
    if k in globals():
        v=globals()[k]
        try:
            a=np.asarray(v); print(k, a.shape, np.round(a.ravel()[:8].astype(float),3))
        except Exception as ex: print(k, type(v))
W0=A(load('z_post.0.weight')); b0=A(load('z_post.0.bias')); W2=A(load('z_post.2.weight'))
for j in [12,88,105]:
    w=W0[j]-A(load_original('z_post.0.weight'))[j]
    print(j,'segnorm ctx/e/u', [round(float(np.linalg.norm(W0[j,s:s+192])),2) for s in (0,192,384)], 'b',round(float(b0[j]),2),
          '|outcol|',round(float(np.linalg.norm(W2[:,j])),3),'top z',np.argsort(-np.abs(W2[:,j]))[:4])
```