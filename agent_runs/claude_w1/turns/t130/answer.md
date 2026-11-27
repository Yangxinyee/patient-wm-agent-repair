This stay gives both drugs in every drug bundle (nA = 2), so its u vectors mix insulin and NE. For a clean dose grid I need to build action-set embeddings directly for {insulin(d)}, {insulin(d), NE(r)} and {NE(r)}. I'll print the signatures of the replica's atom and set-encoder helpers, so I can call them on synthetic atoms.

```python
for f in ['encset','enc_set','enc_set2','encode_set','avec','atom_vec','atom_e_full','atom_e','atoms_of_stay','mlp','mlp2n','f_at']:
    try: print(f, str(inspect.signature(g[f]))[:140])
    except Exception as e: print(f, 'ERR', e)
```