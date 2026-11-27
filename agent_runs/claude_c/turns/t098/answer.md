Run 86 showed I already have state dumps for both the original and an edited model from pass 5 (`fwd_5`). That pass was most likely the x5 numeric-input edit. Those dumps let me find, without a forward pass, where the dose amplification gets lost: the action encoder/z, the state h, or the head. I'll check the npz keys and the edit recorded in the summary.

```python
import json
print(json.load(open('/work/fwd_5/summary.json')))
for f in ['states_original.npz','states_edited.npz']:
    d = np.load('/work/fwd_5/'+f, allow_pickle=True)
    print(f, {k: d[k].shape for k in d.files})
```