The remaining tensors include `filter_gate` (192×384), `filter_norm` and `pending_mlp`. They aren't in my head notes, and "pending" suggests the model represents ongoing actions, such as infusions that have started but not ended. That could be a direct, already-existing route from the current insulin or norepinephrine dose to the prediction, much simpler than writing dose memory into the slots. I'll pull just the ARCHITECTURE.md lines that mention them.

```python
txt = open('/model/ARCHITECTURE.md').read().split('\n')
hits = [i for i,l in enumerate(txt) if any(w in l.lower() for w in ('pending','filter'))]
print("n lines:", len(txt), "hit lines:", hits)
shown = set()
for i in hits:
    for j in range(max(0,i-2), min(len(txt), i+4)):
        if j not in shown:
            shown.add(j); print(j, txt[j][:220])
```