The swap test was not testing what I intended. Its NLL change matched the earlier copy test to four decimals, code by code, and the retired u token still received 50% attention at the 99th percentile.

The cause is in my replica. At bundles with no actions, both the teacher-forced runner and the rollout use a stored constant `u_empty`, not `encode_action(∅)` with the edited bias. So at those bundles the retired token was never shifted, and it tied with its copy.

At drug bundles the swap behaved as designed: the copy won heads 0–2 with weight 0.955. With zero injection the outputs equal the original model's, so the factual fit is preserved either way.

What matters is what the real model does with an empty action set. If it runs `encode_action` (final norm included), my bias shift applies there too. Then the copy (u_empty) wins head 3 at reading bundles, which is also the intended behaviour. If the model uses a separate constant, the tie remains at reading bundles, which is harmless (identical values).

Next I'll check ARCHITECTURE.md for how u is formed when a bundle has no action atoms, and patch the replica's empty-u to `set_enc(W, [], 'encode_action')` if that is how the model does it. Either way the drug-bundle behaviour, which carries the injection, is already correct.

```python
for kw in ['empty','no action','encode_action','actions']:
    for m in list(re.finditer(kw,A_))[:2]: print(kw,'|',A_[max(0,m.start()-150):m.start()+250].replace('\n',' ')); print('..')
```