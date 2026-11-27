Run 99 gives the bin centres. A gf distribution spans about 70–270 mg/dL over 16 bins, and map_arterial spans 33–101 mmHg. A logit tilt along the centres therefore shifts the mean by roughly κ × predictive variance.

Next I want an exact re-implementation of the value head, checked against fwd_4's predicted means for the same 100 stays whose states I have. Several of its inputs are undocumented: 8 register inputs where the text lists 4, the type index, and the gap scaling. A first cheap clue is which `last_numeric` columns and `type_emb` rows are actually used, judging by their weight norms.

```python
ln_w=load('last_numeric.weight'); print('last_numeric col norms',np.linalg.norm(ln_w,axis=0).round(2))
te=load('type_emb.weight'); print('type_emb row norms',np.linalg.norm(te,axis=1).round(2))
print('last_status norms',np.linalg.norm(load('last_status.weight'),axis=1).round(2), 'code norms',np.linalg.norm(load('embed.code.weight'),axis=1).round(1))
print('cnt head bias',load('count_head.bias').round(1))
```