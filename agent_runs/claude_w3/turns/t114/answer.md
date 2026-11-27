My bundle table includes END atoms (kind 2) at each record's end time, carrying the ended record's rate and dose. The dump's atom counts agree: 2 atoms at hours with an infusion start plus the previous END. So atom counts are not the source of the error.

I'll rewrite the encoder and z-posterior check cleanly. To do that I need the exact vocab format: normalisation fields, bin edges, and the index maps for kind, ekind and unit.

```python
for k_, v_ in VOCAB.items():
    s_ = json.dumps(v_)
    print(k_, type(v_).__name__, len(s_), '|', s_[:300])
```