The saved states come from 12 tuning stays, and the rows aren't sorted by stay, so my alignment was fine. The real difference is that my old `zpost_err` built the encodings with `atom_vec` and `set_enc_` rather than `enc_bundle`, and got a 0.10 error. So `enc_bundle` is probably an earlier, less accurate encoder.

Rather than reverse-engineer old code, I'll write a fresh encoder straight from vocab.json, checking each piece. I'll start by printing the vocab keys and the normalisation entries for one code. The goal is to reproduce the z-scores and fine bins exactly.

```python
print(list(VOCAB.keys()))
for k in VOCAB:
    if k not in ('config',):
        v_ = VOCAB[k]; s_ = str(v_)
        print(k, type(v_).__name__, (list(v_.keys())[:12] if isinstance(v_, dict) else ''), s_[:160].replace('\n',' '))
```