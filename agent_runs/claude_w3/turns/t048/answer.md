The tensor groups show the edit options.
- **Action encoder:** `encode_action` is a 192-wide two-MLP set encoder with a norm.
- **Value head:** `attr_heads.value_bin` is a small MLP, 192 → 192 → 17 logits, plus a per-code transition table.
- **Last-value register:** `last_numeric` has 8 inputs.
- **Earlier pass:** fwd_2 holds a factual pass of the original model.

I see two candidate mechanisms:
1. Edit the value head so the glucose and MAP predictions shift with a dose-related feature of the state.
2. Make the action encoding inject a dose signal into h.

Either way, the shift has to respond to the dose relative to what the state already implies; otherwise the factual fit is double-counted. To judge how changes propagate through the rollout, a faithful local copy of the model would help a lot. I can check one against the fwd_3 states and the fwd_2 predictions.

Next I'll print the shapes inside the prelude, blocks and coda, plus a few head and embedding tensors, to see whether a reimplementation is feasible. I'll also confirm that fwd_2 used the sel100 stays. If the shapes match standard transformer layers, I'll attempt the reimplementation; if not, I'll go for a head-only edit.

```python
def shp(prefix):
    return [(x[len(prefix):], SHAPES[x]) for x in NAMES if x.startswith(prefix)]
print(shp('core.prelude.0'))
print(shp('core.blocks.0.0'))
print(shp('core.coda.0'))
print(shp('core.age'), shp('gap'), shp('z_prior'), shp('z_post'))
print([(x,SHAPES[x]) for x in ['last_numeric.weight','type_emb.weight','embed.numeric.weight','embed.kq.weight','embed.text.weight','last_status.weight','embed.categorical.unit','embed.code.weight']])
s2 = json.load(open('/work/fwd_2/summary.json')); print({k_:s2[k_] for k_ in s2 if k_!='results'})
print(VOCAB['config'])
```