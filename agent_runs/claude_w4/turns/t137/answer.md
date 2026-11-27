The kernel can't show these functions' source code. Their call signatures should still be available, so I'll print those. That tells me which helpers I can reuse to rebuild e_k at reading bundles.

```python
for f in ['e_from_atoms_batch','meas_atom','meas_parts','enc_e','set_enc','set_enc2','e_drug_bundle','u_bundle','ctx_of','zmean','build_edit','u_from_atoms','end_atom','atom_emb']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as ex: print(f, 'err', ex)
```