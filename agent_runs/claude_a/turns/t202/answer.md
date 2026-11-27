The replica interfaces are confirmed:
- `prelude()` returns the per-head attention weights and values (`att`, `vh`, `o`), plus `n3` and `E`.
- `core_rest(..., keep=True)` returns the block-0 FFN inputs `n3_0`.

So the staging decoder and the write gate can both be fitted offline. Both atom encoders have a 768×192 first layer, so cutting a gate dim from them means zeroing one input column in each.

A dataset built by `build_ds` probably holds e and u precomputed with the original encoders. If so, an encoder edit would not show up in `tf_eval` unless I re-encode the bundles.

Next I will check:
- the keys and shapes of my cached datasets dsE and dsF;
- the signatures of the encoder helpers (`atom_vectors`, `encode_bundles`, `encode_set`, `deepsets`, `make_atoms`);
- whether the encoder used is a global dict.

This decides whether I can evaluate the code-gate encoder cut directly or must re-encode first. If the ds stores e and u as arrays, I will re-encode with an explicit DeepSets replica that takes the params dict.

```python
for f in ['atom_vectors','atomvec1','encode_bundles','encode_set','deepsets','make_atoms','tf_states_ds','prep_stay','tf_predict']:
    try: print(f, str(inspect.signature(globals()[f]))[:120])
    except Exception as ex: print(f,'ERR',ex)
for nm in ['dsE','dsF']:
    d_=globals().get(nm)
    if d_ is None: print(nm,'missing'); continue
    print(nm, {k:(tuple(v.shape) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else v)) for k,v in d_.items()})
```