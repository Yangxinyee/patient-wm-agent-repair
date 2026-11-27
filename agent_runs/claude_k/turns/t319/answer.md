Pass 15 shows the norepinephrine reader is far too costly in factual fit:

| Readings | Original NLL | Edited NLL (κ = 1.5) | Change |
|---|---|---|---|
| All values | 1.4852 | 1.5191 | +0.034 (limit 0.005) |
| Glucose | — | — | ~0 |
| map_arterial | — | — | +0.044 (MAE 4.10 → 4.20) |
| map_cuff | — | — | +0.083 (MAE 6.54 → 6.91) |

- **Gate:** glucose is untouched, so the gate is clean.
- **Size:** the cost is three to four times what pure reader noise (natural sd 0.69) should produce. That suggests part of it may be a systematic offset in the reader's MAP shift, not just noise.

Next I analyse the paired factual files offline, with no forward pass. For each MAP code I compute three things:
- the shift in predicted mean (edited minus original), its mean and sd;
- its correlation with the original model's error;
- a fit of the per-reading NLL change against a Gaussian-tilt model, used to predict the NLL change at other gains and after re-centring the shift.

I also check the layout of the compensation tensor and the gate biases, to see how a per-code re-centring could be written.

How to read the result:

| Outcome | Conclusion | Next step |
|---|---|---|
| Large mean shift, and re-centring predicts a much lower cost | Offset hypothesis confirmed | Re-centre, then rescale κ |
| Shift centred, cost purely ∝ κ² | Offset hypothesis refuted | Cut κ to about 0.4, or drop the reader |

```python
import numpy as np, pandas as pd
_fo = pd.read_csv('/work/fwd_15/factual_original.csv'); _fe = pd.read_csv('/work/fwd_15/factual_edited.csv')
_cols = list(_fo.columns)
def _pk(keys, excl=()):
    for c in _cols:
        cl = c.lower()
        if any(k in cl for k in keys) and not any(e in cl for e in excl):
            return c
    return None
_cev = _pk(['event', 'code', 'kind', 'name']); _cmu = _pk(['mean'], excl=['abs']); _cnl = _pk(['nll'])
_cy = _pk(['value', 'obs', 'true', 'actual'], excl=['pred', 'nll', 'mean', 'bin'])
print('cols', _cols, 'rows', len(_fo), len(_fe), '| picked', _cev, _cmu, _cnl, _cy)
_res293 = {}
_fl = [0.25, 0.5, 0.75, 1.0, 1.5]
try:
    _key = [c for c in _cols if any(k in c.lower() for k in ['stay', 'time', 'hour'])]
    print('aligned', all(np.array_equal(_fo[c].values, _fe[c].values) for c in _key + [_cev]))
    _tot = {f: 0.0 for f in _fl}; _tot0 = 0.0
    for ev in ['map_arterial', 'map_cuff', 'glucose_finger']:
        mm = (_fo[_cev].astype(str) == ev).values
        mo = _fo[_cmu].values[mm].astype(float); me = _fe[_cmu].values[mm].astype(float); yy = _fo[_cy].values[mm].astype(float)
        dd = me - mo; rr = yy - mo; dn = (_fe[_cnl].values[mm] - _fo[_cnl].values[mm]).astype(float)
        xx = dd**2 - 2*rr*dd
        kk = float((xx*dn).sum()/max((xx*xx).sum(), 1e-12)); db = float(dd.mean()); ss = dd - db
        fc = float(np.corrcoef(xx, dn)[0, 1]) if dd.std() > 1e-9 else 0.0
        crr = float(np.corrcoef(dd, rr)[0, 1]) if dd.std() > 1e-9 else 0.0
        fopt = float((rr*ss).mean()/max((ss*ss).mean(), 1e-12))
        pr = {f: float(np.mean(((f*ss)**2 - 2*rr*f*ss)*kk)) for f in _fl}
        _res293[ev] = dict(n=int(mm.sum()), kk=kk, db=db, sd=float(dd.std()), fopt=fopt)
        if ev.startswith('map'):
            for f in _fl: _tot[f] += pr[f]*mm.sum()/len(_fo)
            _tot0 += dn.sum()/len(_fo)
        print(ev, 'n', int(mm.sum()), 'd mean %.3f sd %.3f corr(d,res) %.3f meanres %.3f | dNLL obs %.4f k2v %.4f fitcorr %.2f | cen pred %s | fopt %.2f' % (
            db, dd.std(), crr, rr.mean(), dn.mean(), kk, fc, ' '.join('%g:%.4f' % (f, pr[f]) for f in _fl), fopt))
    print('overall MAP part obs %.4f | recentred pred per-reading: %s' % (_tot0, ' '.join('%g:%.4f' % (f, _tot[f]) for f in _fl)))
except Exception as ex:
    print('ERR', repr(ex)[:300])
try:
    _et = np.asarray(load('embed_transition.value_bin.weight')); _eto = np.asarray(load_original('embed_transition.value_bin.weight'))
    _ch = np.abs(_et - _eto) > 0
    print('ET', _et.shape, 'chg rows', np.where(_ch.any(1))[0][:12].tolist(), 'chg cols', np.where(_ch.any(0))[0][:12].tolist())
    _b0 = np.asarray(load('attr_heads.value_bin.0.bias')); _b0o = np.asarray(load_original('attr_heads.value_bin.0.bias'))
    print('b0 120 %.3f (orig %.3f) 110 %.3f (orig %.3f)' % (_b0[120], _b0o[120], _b0[110], _b0o[110]))
except Exception as ex:
    print('ERR2', repr(ex)[:300])
with open('/work/notes.md', 'a') as fh:
    fh.write("- fwd14 dose_edit both tuning100 (NE reader kappa1.5): ins x2 -2.12 (orig -2.15), half 1.25; NE x2 1.46 (orig .34), half -.225 (orig -.057): reader works but asymmetric (half only 15% of x2 contribution; model's own NE response equally convex).\n"
             "- fwd15 factual both tuning100: value NLL 1.4852 -> 1.5191 (+.034, limit .005!): map_art +.044, map_cuff +.083, glucose 0. run293: offline decomposition of cost (offset vs noise) from fwd15 csvs.\n")
```