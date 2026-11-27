The helper sources cannot be recovered; only the function objects survive. The state file has the 5,075 bundle rows of setA, each with its stay, hour, drug flags and atom count, but no previous-state links or exact times.

So I will rebuild a clean, documented pipeline in a few runs:

1. Parse the records into bundles.
2. Encode the atoms with the rules recorded in my notebook.
3. Verify the encoding by reproducing the tool's latent posterior means. This was exact before, so it is a strong check that the rebuild is right.
4. Only then test the core step.

To avoid guessing, I first collect the reference facts: the data README, the first rows of the tuning file, the embedding table names and shapes, and the signatures of my surviving helpers. Some helpers may still be callable.

What would let me move straight to the core step: the README and the embedding tables match my notebook's encoding rules (codes, units, kinds, bins). What would need fixing first: a mismatch, such as an extra categorical table.

```python
print(open('/data/README.md').read()[:3000])
import pandas as pd
tu = pd.read_csv('/data/tuning.csv')
print(tu.shape); print(tu.head(12).to_string())
for n in NAMES:
    if n.startswith('embed.'): print(n, SHAPES[n])
for fn in ['LN','mlp','lin','enc_set','enc_set2','encode_set','atom_e_full','atom_e','atom_e_unit','atom_vec','head_P','head_P2','block_stats','deconv','lp_series','ne_expect','add_pol']:
    obj = globals().get(fn)
    if callable(obj):
        try: print(fn, inspect.signature(obj))
        except Exception as ex: print(fn, 'sig?', ex)
print(list(VOCAB.keys()))
```