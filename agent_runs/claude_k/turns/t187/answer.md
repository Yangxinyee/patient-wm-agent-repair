Found it. `view_from` reads the global `T`, which used to hold all 181 tensors. In run 162 I overwrote `T` with only the embedding tables, which broke `view_from`. `datom` only needs the embedding keys, so restoring the full dict is safe.

I'll restore `T` and redo the check on the first readings. The state row is `X8o[kidx]`, and dt runs from that bundle to the reading's hour. If the rebuilt q matches `Q99` to ~1e-5, I can compute q for both models on all 5521 readings and move on to the propagation analysis.

```python
T={n_:np.asarray(load_original(n_),dtype=np.float64) for n_ in NAMES}
print('T restored', len(T))
hr_read=None
for cand_ in ['hr','hrc','hr_r']:
    v_=globals().get(cand_)
    if v_ is not None and hasattr(v_,'shape') and v_.shape==(5521,): print(cand_, v_[:6])
for r0_ in [0,2,5]:
    kp=kidx[r0_]
    for hh_ in [np.ceil(hr8[kp]+1e-6), np.floor(hr8[kp])+1]:
        dts_=(hh_-hr8[kp])*3600.0
        qv=np.asarray(view_from(X8o[kp], dts_)).ravel()
        print('reading',r0_,'prev hr',hr8[kp],'read hr',hh_,'err',np.round(np.abs(qv-Q99[r0_]).max(),6))
```