The code table is longer than the excerpt. Next I'll print just the id-to-name pairs and the action flags compactly, then find this stay's bundles that contain drug atoms.

```python
print([(c['id'],c['name'][:14],c.get('is_action')) for c in codes][5:])
cid={c['name']:c['id'] for c in codes}
acts=[i for i,bb in enumerate(B) if bb['code'].isin([cid.get('insulin'),cid.get('norepinephrine')]).any()]
print(len(acts), acts[:8]); 
for i in acts[:3]: print(B[i].values.tolist())
```