Linear decodability of the dose from the full `norm1(u)` is only R² 0.85 (insulin) and 0.89 (NE). The error grows at 2–2.5× doses, so the u-token representation saturates. Since the head input carries almost no dose-caused signal, the only workable route is a dedicated register:
1. A repurposed head-2 value dimension writes the decoded dose into one state channel.
2. That channel is shielded from the rest of the network (input columns zeroed, other writes blocked).
3. `b_in` routes it from E to the state.
4. A value-head unit reads it and tilts the glucose/MAP bins.

To plan which tensors must be shielded and pick low-importance channels, I need the remaining tensor names: blocks, coda, prior/posterior, flow, and the head norms. As a first ranking, I'll also compute the replica NLL cost of blinding the value head to each input channel.

```python
grp_names = [nm for nm in NAMES if not nm.startswith(('core.prelude', 'embed.', 'attr_heads.', 'embed_transition'))
             and not any(t in nm for t in ['norm1.', 'norm2.', 'norm3.'])]
print([(nm.replace('core.', 'c.').replace('weight', 'W').replace('bias', 'b'), SHAPES[nm]) for nm in grp_names][:60])
costk = np.zeros(192)
for k_ in range(192):
    W1k = W1.copy(); W1k[:, k_] = 0
    Lk = gelu(Xf @ W1k.T + b1) @ W2.T + b2 + TR[cid4[cf]*17 + lbf]
    costk[k_] = (nll_from(Lk) - base_nll).mean()
ok_ = np.argsort(costk); print("value-head input channels cheapest to blind", ok_[:10], np.round(costk[ok_[:10]], 5))
```