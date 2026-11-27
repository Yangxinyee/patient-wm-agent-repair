On these 100 tuning stays, my truth estimates against the edited model:

| Readout | Truth | Edited |
|---|---|---|
| insulin ×2 | −13.38 | −9.85 |
| insulin ×½ | +6.69 | +4.13 |
| NE ×2 | +4.89 | −0.80 |
| NE ×½ | −2.45 | −1.91 |

The profiles show that the insulin correction gets the first hour right (−3.35 vs −3.00). After that it ramps roughly linearly where the truth front-loads, and it only catches up around hour 11. Only the mean over stay-hours is judged, so a single gain that brings the mean to truth may be enough.

The edited x2:½ ratio of added response is 2.4 against 2.0 in the truth, so the edit responds more to increases than decreases. Scaling linearly, insulin ×2 needs a_ins ≈ −2.7, which will raise the factual cost.

NE ×2 fails completely. My hypothesis is that doubled NE rates (>1.0) land in value bins never seen in training (13–16), so the state is out of distribution. If the input embedding has per-code bin entries, I could set NE bins 13–16 to an extrapolation of the trained bins at zero factual cost.

This run therefore does two things. It lists the embedding tensors and the ARCHITECTURE lines on how values are embedded, to see if NE bins have their own rows. It also runs one offline `eval_edit` of the current edit so I learn its output format and can then sweep cost against a_ins.

```python
emb = [(n, SHAPES[n]) for n in NAMES if 'embed' in n or 'bin' in n.lower()]
print(emb)
arch = open('/model/ARCHITECTURE.md').read()
lines_ = arch.split('\n')
hits_ = [i for i,l in enumerate(lines_) if ('bin' in l.lower() or 'numeric' in l.lower() or 'atom' in l.lower()) ]
print(len(lines_), hits_[:60])
shown=set()
for i in hits_[:40]:
    for j in range(max(0,i-1), min(len(lines_), i+2)):
        if j not in shown:
            shown.add(j); print(j, lines_[j][:220])
r_ = eval_edit(gated3(-1.9, 3), 'g3 -1.9/3', True)
print(type(r_), (list(r_.keys()) if isinstance(r_, dict) else (len(r_) if hasattr(r_,'__len__') else r_)))
```