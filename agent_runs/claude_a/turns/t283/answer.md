With re-encoding, the write side works. The memory tracks the ideal 2.5 h-decayed sum of insulin start rates with correlation 0.953 (slope 0.31, residual sd 0.17). Factual cost is −0.0022 on sel2 and −0.0007 on dsE, so it still helps slightly. The memory values are small (sd 0.56), which keeps layer-norm disturbance negligible.

Next comes the readout: a reused value-MLP unit, switched on only for glucose_finger. It will read the difference between the memory dim and the zero reference dim after the head's layer norms, with the linearly predictable part removed so the factual fit is preserved. Before building it I'll inspect the relevant head tensors:
- names and shapes of the head tensors;
- the code index for glucose_finger;
- the current layer-norm, code-embedding and input weights at dims 50 and 99.

That shows what the ablation left in place and what I must set.

```python
hn_=[(k,tuple(v)) for k,v in SHAPES.items() if not k.startswith('core') and not k.startswith('encode') and not k.startswith('z_')]
print('head tensors:', ', '.join('%s%s'%(k.replace('weight','w').replace('bias','b'),list(s)) for k,s in hn_)[:900])
codes=VOCAB['codes']; print('codes', (list(codes.items())[:10] if isinstance(codes,dict) else codes[:10]))
for nm_ in ['out_norm.weight','out_norm.bias','query_norm.weight','query_norm.bias']:
    if nm_ in PA: print(nm_, [round(float(PA[nm_][j]),3) for j in (50,99,79)], 'orig', [round(float(PO[nm_][j]),3) for j in (50,99)])
W0A=PA['attr_heads.value_bin.0.weight']; print('W0 cols 50/99 abs sum', float(W0A[:,50].abs().sum()), float(W0A[:,99].abs().sum()), 'hidden', W0A.shape[0])
ce=[k for k in PA if 'code' in k and 'embed' in k]; print('code emb tensors', ce, [tuple(PA[k].shape) for k in ce])
```