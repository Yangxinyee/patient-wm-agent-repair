`prelude2` gives the exact prelude internals at drug bundles. Each slot token's pre-norm residual (x1) is normalised by norm3 (n3), feeds 768 GELU units (pre/hid), and is then added back (x2) before `prelude_norm`.

I'll build the injection inside this FFN: gated unit pairs that add δ × (insulin rate − expected rate given state). That needs linear readouts from n3 for an insulin-present gate, with a wide margin including at measurement bundles, and for the dose surprise. Before fitting them I need the per-bundle labels already in the kernel: insulin and NE presence, rates, and the latest glucose. I'll check those arrays and the measurement-bundle tokens TOKn.

```python
for nm in ['TOKn','hasI_d','hasN_d','RI_d','RN_d','G3','Y3','S3','X3','P3','hr3','dr3','dt3','drug_rows4','hasI','hasN','inI','inN','band_np','gl','Yg','ins_r','base_rate','ctxs','idx_map','S_idx']:
    v = globals().get(nm, None)
    if v is None: continue
    sh = tuple(v.shape) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v)
    extra = ''
    if hasattr(v,'dtype') and getattr(v,'ndim',0)==1 and len(v)>0:
        try: extra = 'mean %.3f min %.3f max %.3f' % (float(np.mean(np.asarray(v,dtype=float))), float(np.min(np.asarray(v,dtype=float))), float(np.max(np.asarray(v,dtype=float))))
        except Exception: pass
    print(nm, type(v).__name__, sh, extra)
```