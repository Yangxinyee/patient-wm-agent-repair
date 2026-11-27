Run 70 printed the full tensor list. Its structure matches ARCHITECTURE.md: pre-norm attention and FFN layers, a cross-attention to memory in the two recurrent layers, and a single coda layer. It also shows a few details my rebuild has to get right, for example that `last_numeric` takes 8 inputs, not the 4 named in the document.

Since my earlier code can't be recovered, I'll rebuild the forward pass and check it piece by piece against the recorded states:
1. Check the z posterior first, using the recorded previous state together with my own bundle encodings. This tests the embeddings and set encoders on their own.
2. Then check one core step from the recorded previous h and z. This tests the prelude, the recurrent block and the coda.

Splitting it this way shows exactly where the 8.7% mismatch comes from. Before that, I need the layout of the states file, of the parsed bundles, and of the normalisation and bin tables.

```python
print('Z files:', {k_: Z[k_].shape for k_ in Z.files})
print('bundles_B:', type(bundles_B), len(bundles_B))
k0_ = list(bundles_B.keys())[0]
print('key', k0_, type(bundles_B[k0_]), len(bundles_B[k0_]))
for b_ in bundles_B[k0_][:3]:
    print(b_)
print('codes', VOCAB['codes'])
print('event_kinds', VOCAB['event_kinds'], 'atom_kinds', VOCAB['atom_kinds'])
print('numeric_note', VOCAB['numeric_note'])
print('value_bin_note', VOCAB['value_bin_note'])
num_ = VOCAB['numeric']
print(type(num_), list(num_.keys())[:20] if isinstance(num_, dict) else num_[:2])
print(json.dumps(num_)[:2500])
```