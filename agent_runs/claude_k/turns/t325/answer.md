Set B shows a higher factual cost than set A predicted, and the excess sits on the cuff readings:

| Set B, κ = 0.4 | NLL change |
|---|---|
| All values | +0.0035 (1.6953 → 1.6989) |
| map_cuff (1765 readings) | +0.0087 |
| map_arterial (896 readings) | +0.0005 |
| Glucose | none |

- **Why it is higher:** set B has twice as many cuff readings as arterial ones. Cuff readings carry most of the reader's noise, because the shared tilt shifts the wider cuff predictions about 2.4 times more per reader unit.
- **Pooled estimate:** across sets A and B, κ = 0.4 costs about 0.0026 per value.
- **Spread across stays:** a cuff-heavy set of stays could reach about 0.0035 to 0.004, which is uncomfortably close to the 0.005 limit.
- **Gain at stake:** cutting κ from 0.4 to 0.3 loses only about 0.07 mmHg of norepinephrine ×2 response. It cuts the cost by about 45%, to roughly 0.0015 to 0.002 even for a cuff-heavy mix. That margin is worth more than the small gain.

What I considered and rejected:
- **Re-centring the cuff offset:** it would not lower the expected cost. It would only stop the cost from swinging with each stay set's mean error.
- **Separate cuff gating:** it would need two more repurposed units and a delicate gate, which I cannot test properly with 5 Python runs left.

So I will set κ = 0.3. I will also record the set-B shift statistics for the notes, then re-check the factual cost on both stay sets.

| Re-check result | Conclusion |
|---|---|
| About 0.0012–0.0017 on set A and 0.0018–0.0022 on set B | Plan confirmed: cost is about quadratic in κ, reader stays at 0.3 |
| Clearly more | Reader is not worth keeping; remove it |

```python
_set_kappa_ne(0.3)
_w = np.asarray(load('attr_heads.value_bin.2.weight'))
_fo18 = pd.read_csv('/work/fwd_18/factual_original.csv'); _fe18 = pd.read_csv('/work/fwd_18/factual_edited.csv')
for _ev in ['map_arterial', 'map_cuff']:
    _m = (_fo18['event'] == _ev).values
    _d = _fe18['predicted_mean'].values[_m] - _fo18['predicted_mean'].values[_m]
    _r = _fo18['observed'].values[_m] - _fo18['predicted_mean'].values[_m]
    print('setB k.4', _ev, 'n', int(_m.sum()), 'd mean %.3f sd %.3f corr(d,res) %.3f meanres %.3f' % (_d.mean(), _d.std(), np.corrcoef(_d, _r)[0, 1], _r.mean()))
print('kappa .3 col120 norm %.3f' % np.linalg.norm(_w[:, 120]))
with open('/work/notes.md', 'a') as fh:
    fh.write("- fwd18 factual both set B (100 fresh tuning ids 2020000001..146 not in A) kappa.4: +.0035 (cuff +.0087 n1765, art +.0005 n896, glucose 0): cuff-heavy set -> cost higher. "
             "Pooled est kappa.4 ~.0026, cuff-heavy worst ~.004 -> too close to .005.\n- run296: kappa -> 0.3 (expected cost ~.0015-.002; NE x2 ~.55, half ~-.10). NEXT: factual both on A and B, final dose_edit A, then FINAL.\n")
```