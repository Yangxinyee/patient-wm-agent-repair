The outputs give me what I need to rebuild the head offline:
- states (`x` is out, the 4 h slots and the z posterior mean) for 5,192 bundles;
- factual predicted means for 5,138 readings on the same 100 training stays.

The head-only tensors are the z prior MLP, `z_head`, `gap`, `type_emb`, `query_norm`, the value-head MLP, `last_numeric` (8 inputs), `last_status` and `embed_transition.value_bin` (13 codes × 17 rows). Next I'll read the vocab keys, config, code and type lists, and normalisation and bin tables. A reproduction can only be exact if the type index and register features are right, and I suspect those are what went wrong in run 107.

```python
V = VOCAB
print(list(V.keys()))
for k in V:
    v = V[k]
    s = json.dumps(v)
    print('==', k, type(v).__name__, len(s), s[:700])
```