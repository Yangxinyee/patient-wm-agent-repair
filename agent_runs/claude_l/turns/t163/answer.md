The head tensors are all present, but three details are unknown. These are which `type_emb` row a glucose_finger value query uses (there are 15 types), what the 8 inputs of `last_numeric` are (the doc lists 4), and the exact gap input of the view. The vocab file probably defines the type list and the register features. I'll print its keys and config before guessing.

```python
print(list(VOCAB.keys()))
print({k:v for k,v in VOCAB['config'].items()} if 'config' in VOCAB else '')
for k in VOCAB:
    if k not in ('codes','numeric','value_bin_centres','config'):
        v=VOCAB[k]; print(k, (v if len(str(v))<400 else str(v)[:400]))
print([ (c['id'],c['name']) for c in VOCAB['codes']])
```