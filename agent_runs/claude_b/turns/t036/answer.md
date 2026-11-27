The save failed only because `load` returns torch tensors, which use `.clone()` rather than `.copy()`. I will make the same three-tensor blinding edit with torch operations and check exactly which entries changed.

```python
import torch
W = load_original('embed.numeric.weight').clone()
W[:,2] = 0.0; W[:,4] = 0.0
save_edit('embed.numeric.weight', W)
for n in ['embed.bins.dose.weight','embed.bins.rate.weight']:
    T = load_original(n).clone()
    T[1:17] = T[8].unsqueeze(0)
    save_edit(n, T)
print(edited())
for n in edited():
    a = load(n); b = load_original(n)
    print(n, tuple(a.shape), "changed entries:", int((a!=b).sum()), " max abs change", float((a-b).abs().max()))
json.dump([int(s) for s in sel1], open('/work/sel1.json','w'))
```