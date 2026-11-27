The old helpers are still callable and their signatures survive. That should save many runs. For example, `atom_e_full(mg_row, units_row)` matches my note "use atom_e_full(4,3)", and `atom_vec`, `encode_set` and `LN` look like the encoder building blocks.

The data format is now clear:
- Events have times in hours.
- Infusion rows carry a rate, a dose and an end time.
- Measurements carry values.

Next I will:
- list the remaining helper names (the earlier listing stopped at "q_"), to find the latent-posterior and context functions;
- print the vocabulary's normalisation and bin notes;
- probe what `atom_e_full(4,3)` returns.

What decides the next step:
- **Reuse:** the helpers include a posterior function and a bundle builder. Then I can verify one stay's latent means against the tool's saved states right away.
- **Rewrite:** they do not, and I write a compact bundle builder next.

```python
rest = sorted(k for k,v in globals().items() if not k.startswith('_') and k > 'q_' and (callable(v) or hasattr(v,'shape') or isinstance(v,(dict,list))))
print(rest)
for k in ['numeric','numeric_note','value_bin_note','codes','event_kinds','atom_kinds']:
    print(k, '=>', str(VOCAB[k])[:700])
r = atom_e_full(4,3)
print(type(r), (r.shape if hasattr(r,'shape') else str(r)[:300]))
for fn in rest:
    obj = globals()[fn]
    if callable(obj) and not isinstance(obj, type):
        try: print(fn, inspect.signature(obj))
        except Exception: pass
```