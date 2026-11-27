Two findings.

**Bin embeddings are shared across codes.** The fine-bin tables (`embed.bins.value`, `embed.bins.rate`, `embed.bins.dose`) are 17×192 tables used by every code. Doubled NE rates fall in rate bins 13–16, and insulin uses those bins in training: tier-4 doses above 5 u/h land in bin 16. So editing those rows to fix NE ×2 would change insulin inputs and cost factual fit. The free-edit idea is dead. NE ×2 can only be helped from inside the value head, or by a correction that does not depend on decoding the out-of-range rate.

**`eval_edit` crashed.** Inside `mlp2`, a global lambda is now called with 2 arguments but takes 1, so some helper was probably overwritten in a later run. I'll find which globals are lambdas, what `mlp2` and `ctx_from_state` call, and their arguments, so I can repair it and get the offline cost curve.

```python
lam_ = {k:(v.__code__.co_argcount, v.__code__.co_varnames[:v.__code__.co_argcount]) for k,v in list(globals().items()) if callable(v) and getattr(v,'__name__','')=='<lambda>'}
print('lambdas:', lam_)
for nm in ['mlp2','mlp','ctx_from_state','q_from_rows','reg_feats','value_logits','T','gelu_','gelu','gel','act_fn','ln','lnorm','lin','set_encode','from_src']:
    fo = globals().get(nm)
    if fo is None: print(nm,'MISSING'); continue
    if not hasattr(fo,'__code__'): print(nm, type(fo)); continue
    co = fo.__code__
    print(nm, co.co_varnames[:co.co_argcount], '| names:', co.co_names[:25])
print(type(globals().get('act')), type(globals().get('ACT')), type(globals().get('ACTS')))
```