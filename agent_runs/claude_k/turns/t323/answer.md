Pass 17 confirms the norepinephrine reader's rollout response scales linearly with its gain.

| Edit | Original | Edited, κ = 1.5 | Edited, κ = 0.5 |
|---|---|---|---|
| Norepinephrine ×2 | +0.336 | +1.457 | +0.698 |
| Norepinephrine ×½ | −0.057 | −0.225 | −0.126 |
| Insulin ×2 | −2.148 | −2.118 | −2.139 |
| Insulin ×½ | +1.227 | +1.251 | +1.224 |

- **×2 contribution:** the reader adds +0.36 at κ = 0.5. Linear scaling from κ = 1.5 predicted +0.37.
- **×½ contribution:** the reader adds −0.07, against −0.06 predicted.
- **Insulin:** unchanged, as intended, since the gate stays off for glucose codes.

The two factual passes fit a cost curve of 0.0376 f² − 0.0037 f, where f = κ / 1.5:

| κ | Expected factual cost (nats per value) | Norepinephrine ×2 |
|---|---|---|
| 0.5 | 0.0029 | +0.70 |
| 0.4 | about 0.0017 | about +0.63 |

At κ = 0.5 the tuning estimate has a standard error of about 0.0013. That leaves a few-percent chance of breaking the 0.005 limit on unseen stays, which would forfeit everything. The gain over κ = 0.4 is only 0.07 mmHg. So κ = 0.4 is the better trade.

Next steps:
- Set κ = 0.4.
- List 100 tuning stays that were not used so far.
- Check the factual cost on those fresh stays as an independent test.

If that cost comes out at about 0.002 or less, κ = 0.4 is confirmed. If it comes out above about 0.003, I will reduce κ again or remove the reader.

```python
_set_kappa_ne(0.4)
_w = np.asarray(load('attr_heads.value_bin.2.weight'))
_A295 = set(int(x) for x in _ids13)
_t295 = [int(x) for x in _tids292]
print('kappa .4 col120 norm %.3f | tuning contiguous %s' % (np.linalg.norm(_w[:, 120]), _t295 == list(range(_t295[0], _t295[0] + len(_t295)))))
_B295 = [x for x in _t295 if x not in _A295][:100]
print('IDSB=' + json.dumps(_B295, separators=(',', ':')))
with open('/work/notes.md', 'a') as fh:
    fh.write("- fwd16 factual both kappa.5: +.0029 (art +.0047, cuff +.0062). fwd17 dose_edit kappa.5: NE x2 .698 (contrib +.362, linear in kappa), half -.126, insulin unchanged. "
             "Cost fit ~ .0376 f^2 - .0037 f (f=kappa/1.5).\n- run295: kappa -> 0.4 (pred cost .0017, NE x2 ~.63). NEXT: factual both on fresh tuning set B (first 100 tuning ids not in A) to verify cost.\n")
```