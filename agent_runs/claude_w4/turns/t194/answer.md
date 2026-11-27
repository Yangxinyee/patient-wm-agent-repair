`prelude2` returns a dict. The prelude layer is pre-norm attention (q, kv, out, with no biases), then norm3 and a two-layer FFN, then `prelude_norm`. I'll list the dict's contents and the FFN hidden size, to find the pre-norm residual x and the FFN pre-activations at drug bundles.

```python
r2 = prelude2(TOK, W_)
for k,v in r2.items(): print(k, tuple(v.shape) if hasattr(v,'shape') else type(v).__name__)
print('ffn0', SHAPES['core.prelude.0.ffn.0.weight'], 'ffn2', SHAPES['core.prelude.0.ffn.2.weight'])
```