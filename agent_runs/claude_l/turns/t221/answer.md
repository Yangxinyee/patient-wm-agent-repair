The probe shows the mechanism. The predicted glucose response to a z-token displacement a × (insulin dose tangent) is a **step**, not a slope:

| a | −40 … −5 | 0 | +5 … +40 |
|---|---|---|---|
| Change in next-hour finger glucose | 0 | 0 | ≈ −0.9 (sd ≈ 1) |

The prelude attention is effectively hard. When the z token is a copy of u, the two tokens tie. Any displacement that lowers the z token's attention score leaves u attended alone, so nothing happens. Any displacement that raises it makes z win outright, so the slots read z's value, and here it saturates almost at once because 5 × |tangent| ≈ 49 already dwarfs |u|.

That explains both earlier results: κ = 4 acted as a rectified, saturated switch on the residual's sign. So the injection has to be redesigned:
1. **Booster:** at drug bundles only, a small "booster" component makes the z token always win in the u-attending heads (0–2) by a safe margin. It must barely change the value read and must not make head 3 leave its slot tokens.
2. **Residual direction:** the residual is injected along directions that leave the attention scores unchanged but change the value. The response is then linear and symmetric in the residual.
3. **Reading bundles:** z stays an exact copy of u_empty, a perfect tie, so there is no change at all.

To design this I need the attention scores and their sensitivity. Next I'll check what my existing `prelude_att` helper returns, and measure at one drug bundle the per-head attention weights on each token and the score gap between u and the runner-up token.

```python
try: print('prelude_att', inspect.signature(prelude_att))
except Exception as ex: print(ex)
for n in NAMES:
    if 'prelude' in n: print(n, SHAPES[n])
```