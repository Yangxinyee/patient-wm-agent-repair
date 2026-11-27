`w_ins` and `w_ne` are not decoders (their lengths are 28,990 and 41,748), so I have to rebuild the u-token dose decoders. `bundles_of_stay` needs the rate-unit map (`RU0` is the correct one) and takes a `dscale` argument, which presumably produces counterfactual-dose bundles. Next I'll inspect one drug bundle to find where its u-token atoms live and how they are encoded. Then I can compute norm1(u) for recorded and rescaled doses and fit a dose decoder that is linear over the range the ×2 edits reach.

```python
B0_ = bundles_of_stay(G3[s0_], RU0)
print(type(B0_).__name__, len(B0_))
def desc(o):
    if isinstance(o, dict): return {k_: desc(v_) for k_, v_ in list(o.items())[:8]}
    if isinstance(o, (list, tuple)): return f"{type(o).__name__}[{len(o)}]" + (f"<{desc(o[0])}>" if len(o) and not isinstance(o[0], (int, float)) else "")
    if isinstance(o, np.ndarray): return f"arr{o.shape}"
    return f"{type(o).__name__}:{str(o)[:20]}"
for k_ in range(min(4, len(B0_))): print(k_, str(desc(B0_[k_]))[:300])
```