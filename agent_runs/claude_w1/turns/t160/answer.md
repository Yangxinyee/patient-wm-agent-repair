The labels field holds strings, which broke my print. The useful result is the signatures of the replica building blocks (mha, ffn, LNn, linn, mlp2n, wc_vec, zpost_of). With them I can assemble any local test I need. This run looks at the bundle fields: time units, the meaning of labels, and the action/measurement counts. It also tries prelude_attn on a drug bundle and on a reading bundle to see the prelude attention pattern. That pattern decides whether a value-projection injection fires only at drug bundles.

```python
for k in range(1, 9):
    b = B0[k]; print(k, 't', np.round(b['t'],4), 'nA', b['nA'], 'nM', b['nM'], 'labels', [str(x)[:25] for x in b['labels']])
for k in [2, 3]:
    try:
        r = prelude_attn(St0[k-1], B0, k)
        if isinstance(r, (tuple, list)): print(k, 'tuple', [np.shape(x) for x in r])
        else: print(k, 'arr', np.shape(r), np.round(np.asarray(r).reshape(-1)[:14], 3))
    except Exception as ex: print(k, 'err', str(ex)[:150])
```