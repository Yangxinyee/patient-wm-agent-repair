The architecture clarifies the noise source. The state update uses the posterior mean, so it is deterministic. Only the head's z is drawn from the prior: `z_prior` gives a mean and log-sd, and z enters through `z_head` into the view before LN_query. A decoder that loads on z-sensitive directions would therefore add noise both to the tool's factual NLL (4 averaged draws) and to the rollouts.

To build a z-draw variant of my with_code function I need to see which globals and constants wc_vec uses. I'll print its code-object names and constants (short).

```python
co = wc_vec.__code__
print('names', co.co_names)
print('consts', [c for c in co.co_consts if not hasattr(c,'co_code')][:30])
print('LN_query-like tensors', [n for n in NAMES if 'query' in n or 'type_emb' in n or 'gap' in n][:12])
```