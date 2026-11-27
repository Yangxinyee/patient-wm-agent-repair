The residual regressions settle the NE question.

- **Insulin:** the original model's glucose residual depends on the dose surprise (−2.23 ± 0.53, consistent with the −2 one-hour effect) and not on the band mean. So insulin's expected rate can be taken from the state's glucose information alone.
- **NE:** the arterial residual depends on the hour-to-hour rate change (+3.44 ± 0.83, matching the lag-1 effect of 3.7) and also on the rate level itself (+1.52 ± 0.39). A level-based NE injection, using a constant expected rate rather than one built on the state's memory of past rates, therefore costs little factual fit and keeps the full effect of a sustained dose change.
- **Leakage risk:** the state-based expected insulin rate must not respond to the current, edited dose. n3 contains the u_k token's attention output, so a naive ridge would partly read the rate itself.

To enforce that, I'll compute n3 under altered doses with the exact replica, which needs the u-encoder functions. I'll check their signatures.

```python
for nm in ['U_of','u_bundle','u_from_atoms','ins_atom','ne_atom','u_set','enc_set','set_enc2','drug_atom_parts','atom','try_atom']:
    f = globals().get(nm)
    try: print(nm, str(inspect.signature(f))[:120], '|', (f.__doc__ or '')[:80])
    except Exception as ex: print(nm, 'ERR', ex)
```