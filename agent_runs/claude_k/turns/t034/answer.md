The tensor list shows the structure:
- `attr_heads.value_bin` is a two-layer head (192→192→17) on `with_code`, plus a transition table `embed_transition.value_bin` (221×17, i.e. 13 codes × 17 last-value bins).
- Drug atoms enter through `embed.numeric` (192×6 on z_value, has_value, z_dose, has_dose, z_rate, has_rate), `embed.bins.dose/rate`, and `encode_action`.
- `last_numeric` takes 8 inputs, not the 4 the description lists.

Next I'll read tensors.json; it is a dict keyed by tensor name and may describe each tensor's role, activations and input layout. If it documents the remaining inputs and activations, I can work out exactly how a dose flows into the glucose/MAP predictions. If it holds only shapes, I'll have to infer the computation from ARCHITECTURE.md and the state outputs.

```python
for k in list(T.keys())[:8]:
    print(k, json.dumps(T[k])[:400])
print(len(T))
# look for any descriptive fields
ex = T[list(T.keys())[0]]
print(type(ex))
allkeys = set()
for k,v in T.items():
    if isinstance(v, dict): allkeys |= set(v.keys())
print(allkeys)
for n in ['last_numeric.weight','embed.numeric.weight','embed_transition.value_bin.weight','encode_action.size.weight','core.flow.a','filter_gate.weight','embed.kq.weight','type_emb.weight']:
    print(n, json.dumps(T.get(n))[:600])
```