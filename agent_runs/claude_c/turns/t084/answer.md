The last run listed the tensors on the dose route. Every action-encoder tensor (`encode_action.mlp1`, `mlp2`, `size`, `norm`) is used only at drug bundles. Inside the prelude, the action token u_k shares its attention and FFN weights with the slot, reading and z tokens, and no projection exists only for u_k.

So any edit with no side effects at drug-free bundles has to sit upstream of the prelude, in `encode_action` or in the dose/rate input embeddings. The catch is that u_k's final layer norm discards magnitude. That may be why scaling the dose columns 5x gave little.

One more possible route stands out: `last_numeric.weight` has 8 input columns, while the architecture text lists only 4 register features. If the extra columns carry a drug-rate register into the glucose and MAP heads, that would be a direct, dose-specific lever.

This run checks the input meaning of `last_numeric`, `last_status`, `type_emb`, `z_post` and `pending_mlp`, using `tensors.json` and the vocab config. If `tensors.json` gives no descriptions, I'll fall back to the encoder-side levers.

```python
tj = json.load(open('/model/tensors.json'))
print(type(tj), (list(tj.keys())[:5] if isinstance(tj, dict) else tj[:2]))
for k_ in ['last_numeric.weight','last_status.weight','type_emb.weight','z_head.weight','z_in.weight','z_post.0.weight','z_prior.0.weight','pending_mlp.0.weight','filter_gate.weight','query_norm.weight','gap.0.weight','code_bias','embed.code.weight','embed.kind.weight','embed.ekind.weight']:
    if k_ in SHAPES: print(k_, SHAPES[k_], (tj.get(k_) if isinstance(tj, dict) else ''))
print([x for x in NAMES if x.startswith('embed') or 'out_norm' in x or 'flow' in x or 'age' in x or 'gap' in x])
print(json.dumps(VOCAB.get('config', {}))[:1500])
```