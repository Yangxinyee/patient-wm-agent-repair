h's direction is very stable: cosine with the mean is 0.97–0.99 for both raw and layer-normalised h. So a single fixed direction ĥ stands in well for the slot tokens' content. I'll test whether making the action token resemble the slot tokens draws more attention to it, raising the potency of its dose-dependent part. The encoder's output offset now gets a large component along ĥ via a rank-1 update on `encode_action.norm.bias`. Its scale and sign are unknown because the offset's projection on ĥ is unknown, so I start with alpha = 300.

Expected outcomes:
- A large increase in insulin-readout magnitude means attention share is the bottleneck.
- No change means the projection is tiny or the idea is wrong.
- A collapse toward 0 means the offset now swamps the dose signal, so alpha is too large.

```python
clear_ops()
mu_h = np.load('/work/fwd_5/states_original.npz')['x'][:, 192:384].mean(0)
hhat = mu_h / np.linalg.norm(mu_h); np.save('/work/hhat.npy', hhat)
rank1('encode_action.norm.bias', hhat, 300.0)
print(ops()[0]['op'], len(ops()))
```