The hand-built END atoms are much worse than the old `bundle_vecs` encoding (error about 15 vs 0.9), so my `drug_atom2` defaults don't reproduce what `bundle_vecs` does internally. Guessing at my own old helpers isn't productive. Instead I'll rebuild the drug-atom embedding from the vocabulary file, where I can see every field. Next I'll print the vocabulary's structure apart from the bin tables: the kinds, event kinds, categorical tables and units. That should show exactly which embedding rows a START and an END record use.

```python
for k_, v_ in VOCAB.items():
    if k_ in ('numeric',): continue
    s_ = json.dumps(v_)
    print(k_, ':', s_[:700])
```