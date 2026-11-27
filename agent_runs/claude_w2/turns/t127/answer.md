Decompiling `run13` recovers my last verified head computation:
- the view context is out + out_norm(mean of time-flowed h) + gap(log dt/10);
- z_prior's mean goes through z_head and is added together with type_emb[ty];
- the sum passes through query_norm to give q;
- with_code = q + code embedding + last_numeric(8 columns) + bins.value[last bin];
- logits = value MLP + transition-table row, then softmax.

The 8 columns are [has, log(1+age in hours)/10, z_last, has, 0, 0, 0, 0]. It omits the bins.rate and last_status terms and uses the prior mean rather than 4 sampled z's.

Next I inspect fa4, the per-reading factual table from fwd_4 for these stays, to reuse its last-register fields. Then I can compute the head from my exact local states and compare the predicted means with the tool's. A small, unbiased difference that shrinks when averaging over 4 prior draws would confirm the head.

```python
print(list(fa4.columns))
print(fa4.head(3).T.to_string()[:2500])
print(ty if 'ty' in globals() else None, best_ty if 'best_ty' in globals() else None)
```