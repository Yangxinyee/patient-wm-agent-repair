The tensor list shows the pieces relevant to an edit:
- **Value head:** an MLP, `attr_heads.value_bin.0` (192×192) then an activation and `.2` (17×192), reading `with_code`.
- **`with_code` inputs:** the query (`query_norm(view + type_emb)`), the code embedding, and the last-value register. `last_numeric` has 8 inputs, not the 4 in the document.
- **Action path:** `encode_action` is a DeepSets MLP stack ending in a LayerNorm, so dose information reaches the core only as a direction of u_k. From there it goes into `z_post` (576 = 3×192 inputs: ctx, e_k, u_k) and the prelude token.

A small exact edit would be easiest at the output side. That needs a quantity in the state that tracks recent doses, mapped into a shift of the predicted mean.

Before designing anything I need the exact computation: activations, norm types, what the 8 `last_numeric` inputs are, and whether there are biases. tensors.json may describe each tensor, so next I'll print its entries for the key tensors.

```python
for k in ['embed.numeric.weight','last_numeric.weight','attr_heads.value_bin.0.weight','attr_heads.value_bin.2.weight','embed_transition.value_bin.weight',
          'query_norm.weight','z_head.weight','encode_action.mlp1.0.weight','encode_action.norm.weight','z_post.0.weight','core.flow.a','core.slots','type_emb.weight','embed.kq.weight','gap.0.weight']:
    print(k, json.dumps(tj[k])[:700])
print(len(tj))
others = [k for k in tj if k not in NAMES]
print("non-tensor keys:", others[:20])
```