The head source is not recoverable from the kernel, so I will rebuild the value head from ARCHITECTURE.md:
- **view** = ctx + z_head(z), where ctx is the same context the z posterior uses and z is drawn from z_prior(ctx).
- **q** = query_norm(view + type_emb[type]).
- **with_code** = q + code embedding + last_numeric[has_last, log(1+age)/10, z_last, z_last_rate] + the last-bin embeddings + last_status.
- **logits** = the value_bin MLP(with_code) + the transition logit for the last bin.

Some choices are still open: which type row is used, which status index, the age unit, and whether the absent bins' embeddings are added. To settle them I will check against the tool's factual predicted means in fwd_2. First I need the head tensor shapes. Next I will print the head tensor shapes and peek at the fwd_2 factual table.

```python
for n_ in NAMES:
    if n_.split('.')[0] in ('attr_heads','type_emb','last_numeric','last_status','gap','z_head','z_prior','query_norm','embed_transition','z_post') and 'rate_bin' not in n_ and 'status.' not in n_:
        print(n_, SHAPES[n_], end=' | ')
print()
print(F2.head(4).to_string())
print(F2.event.value_counts().to_dict())
```