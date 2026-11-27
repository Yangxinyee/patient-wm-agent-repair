Run 296 shows that set B's extra cost on cuff readings is partly a systematic offset, not only noise.

| Readings (set B, κ = 0.4) | Mean shift | Shift sd | Model's own mean error |
|---|---|---|---|
| map_cuff | −0.50 | 0.90 | +0.48 (under-predicts) |
| map_arterial | +0.05 | 0.47 | — |

- **Cuff offset is systematic:** on set A, scaled to the same κ, the cuff shift was also negative, about −0.27. So the reader is not centred for cuff readings.
- **Why the cost differs between sets:** on B the offset adds to the model's error, while on A it happened to cancel part of it.
- **Arterial:** its offset is negligible.
- **Expected offset at κ = 0.3:** about −0.30 for cuff.
- **Cost of leaving it:** it adds expected cost, and makes the cost on unseen stays depend on their mean error.

The fix is to re-centre the cuff predictions only, by adding a small constant upward tilt to the cuff rows of the per-code logit table.
- **Tilt:** 0.005 per unit of (bin centre − 70), about +0.3 for a predictive variance near 60. That is at most 0.175 logits at the extreme bins.
- **Effect on rollouts:** a constant offset cancels in edited-minus-recorded comparisons, so it should leave every dose-edit shift unchanged and only remove factual cost.
- **Safety check:** I first look up which code is map_cuff in the vocabulary, and confirm that table rows come in blocks of 17 per code. If either check fails, nothing is applied.

What will confirm it: in the next factual passes, the cuff mean shift should be close to 0 on both sets, with costs around 0.0015 to 0.002 and glucose unchanged. If another code's predictions move, the row mapping was wrong, and I will undo this change.

```python
import torch
_codes = VOCAB.get('codes') if isinstance(VOCAB, dict) else None
def _cid(nm):
    if isinstance(_codes, dict):
        if nm in _codes:
            v = _codes[nm]
            if isinstance(v, (int, np.integer)): return int(v)
            if isinstance(v, dict):
                for kk in ('id', 'code', 'index'):
                    if kk in v: return int(v[kk])
        for k, v in _codes.items():
            if v == nm: return int(k)
    if isinstance(_codes, list):
        for i, c in enumerate(_codes):
            if c == nm or (isinstance(c, dict) and c.get('name') == nm): return i
    return None
_cu = _cid('map_cuff'); _ca = _cid('map_arterial')
_ET = np.asarray(load('embed_transition.value_bin.weight')).astype(np.float32).copy()
_ETo = np.asarray(load_original('embed_transition.value_bin.weight')).astype(np.float32)
_chr = np.where(np.abs(_ET - _ETo).max(1) > 0)[0]
print('codes type', type(_codes).__name__, 'arterial', _ca, 'cuff', _cu, '| ET changed blocks', np.unique(_chr // 17).tolist(), 'rows', len(_chr))
_VCm = np.array([35., 41., 46.5, 53., 61., 67.5, 72.5, 76.5, 79.5, 82.5, 86., 89.5, 93.5, 97.5, 100.5, 105.])
if _cu in (5, 7) and _ca in (5, 7) and _cu != _ca:
    _delta = np.zeros(17, np.float32); _delta[1:] = (0.30 / 60.0) * (_VCm - 70.0)
    _ET[17*_cu:17*_cu+17, :] += _delta[None, :]
    save_edit('embed_transition.value_bin.weight', torch.from_numpy(_ET) if _isT292 else _ET)
    print('cuff recentring applied to rows %d-%d, max|delta| %.3f' % (17*_cu, 17*_cu+16, np.abs(_delta).max()))
else:
    print('NOT applied (code lookup failed)')
with open('/work/notes.md', 'a') as fh:
    fh.write("- run296: setB kappa.4 cuff d mean -.50 sd .90 (meanres +.48) -> systematic cuff offset (A: -.27 equiv), arterial ~+.05. kappa set 0.3.\n"
             "- run297: cuff recentring via ET rows 17*cuff..+16: logits bins1-16 += (0.30/60)*(centre-70) (constant tilt, cancels in dose-edit diffs). NEXT: factual both A,B; dose_edit A; FINAL.\n")
```