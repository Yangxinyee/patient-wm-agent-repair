My notebook says an earlier attempt to rebuild the model exactly was set aside as "guesswork": the vocab file has no categorical vocabularies, and the function sources are lost. I don't know how close that attempt got. Before deciding whether rebuilding the core is realistic, I will find the config dict it used and re-run the z posterior check on one stay to measure the mismatch.

- **Small error:** the encoders are right, and only the core layers remain to build.
- **Large error:** rebuilding the core would eat most of my remaining runs, and I should drop the idea.

```python
cands={k:v for k,v in globals().items() if isinstance(v,dict) and not k.startswith('_') and k not in ('NAMES','SHAPES','VOCAB','TJ')}
for k,v in cands.items():
    try: print(k, list(v.keys())[:12])
    except Exception as ex: print(k,'?')
```