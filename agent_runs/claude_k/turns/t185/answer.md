On average the linear decoders are fine. At low ridge strength (|pI| 920, |pN| 375), the mean decoded shift is within about 1% of the true shift for every edit:
- insulin and norepinephrine x0, x½, x2, x3 are all within 1–1.2%;
- the worst case is insulin x1.5, at 0.974.

So the writer can stay a single linear unit per drug, with exact zero on the empty set.

The design problem is now on the reading side. If the writer stores the raw dose, the factual predictions acquire a dose-correlated bias, because doses follow glucose and MAP. The writer should instead store the **dose shock**: dose minus the policy-expected dose, estimated from the context at the drug bundle. That needs a gated design, more z_post units, and a reader in the value head.

Before building all that, I need to know whether a z shift reaches q strongly and cleanly enough to read. fwd_8 already has paired states with known z shifts, so I can estimate this for free:
- Rebuild q at every reading for both models.
- Regress Δq on the z shifts at the preceding drug bundles.
- Compare the response with the factual spread of q along the same direction.

First I need the mapping between `view_from`, `kidx` and `Q99`. I'll test it on a few readings.

```python
print('kidx sample',kidx[:6],'hr of those bundles',hr8[kidx[:6]], 'sid', sid8[kidx[:3]], 'stays_arr', stays_arr[:3] if 'stays_arr' in globals() else None)
print('Q99[0] norm',np.linalg.norm(Q99[0]))
for off_ in [0,1]:
    kprev_=kidx[0]-off_
    if kprev_<0: continue
    # dt from bundle kprev_ to reading bundle: try reading at hour of bundle kidx[0] (or kidx+1)
    for kr_ in [kidx[0], kidx[0]+1]:
        if kr_<len(hr8):
            dts_=(hr8[kr_]-hr8[kprev_])*3600.0
            try:
                qv=np.asarray(view_from(X8o[kprev_], dts_))
                print('prev',kprev_,'reading bundle',kr_,'dt',round(dts_),'err',np.round(np.abs(qv-Q99[0]).max(),5))
            except Exception as ex: print('err',repr(ex)[:200])
```