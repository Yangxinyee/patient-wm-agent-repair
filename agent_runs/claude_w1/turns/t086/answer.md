The rest of the kernel contains the pieces that gave the exact latent-posterior replica: `atom_e_full` (it returns an atom-encoding function), `zpost_batch4`, `zpost_err` and the unit maps `umap` / `rumap`. I can't read their sources.

The vocabulary confirms the encoding rules:
- z = (x − median)/IQR, clipped to [−10, 10];
- fine bin = 1 + the number of edges below the value;
- class 0 means "no value".

Rather than guess how the old helpers expect their inputs, I will inspect their compiled code objects. The variable names, constants and referenced globals and attributes show:
- which embedding tables they add;
- how END atoms are handled;
- how empty sets are encoded;
- what the atom function takes as input.

If the constants show tensor names and an END-atom rule that match my notebook, I will reuse `atom_e_full(4,3)` and `atom_vec` directly in a clean bundle builder. If they don't, I will re-derive the encoder from the documented formula and re-verify it against the saved posterior means.

```python
def show(fn):
    c = fn.__code__
    print('##', fn.__name__, c.co_varnames[:c.co_argcount], '| vars:', c.co_varnames[c.co_argcount:][:30])
    print('   consts:', [k for k in c.co_consts if not hasattr(k,'co_code')][:40])
    print('   names :', c.co_names[:50])
    for k in c.co_consts:
        if hasattr(k,'co_code'):
            print('   inner', k.co_name, k.co_varnames[:k.co_argcount], [x for x in k.co_consts if not hasattr(x,'co_code')][:30], k.co_names[:40])
f_at = atom_e_full(4,3)
for fn in [atom_e_full, f_at, atom_e, atom_vec, encode_set, enc_set2, zpost_batch4]:
    show(fn)
print('umap', umap, 'rumap', rumap)
```