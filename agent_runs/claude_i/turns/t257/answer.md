The architecture notes confirm the update path:

- The z token `z_in(z)` is one of seven tokens in a pre-norm transformer layer, the prelude.
- The four slot outputs pass through `prelude_norm` to give E.
- E feeds `h <- a_bar*h + softplus(delta_in)*b_in(E) + (blocks increment)`.
- Then comes the coda.

Because the prelude is pre-norm, the z token gets normalised before attention. So making z_in's column bigger only changes the z token's direction, not how strongly it shifts the slots. That makes "scale up z_in column 18" a weaker lever than I assumed. The cleanest linear entry point may instead be how E feeds b_in.

Next I'll read lines 30 to 58 in full to see exactly where the residual paths are. A residual path where the z token's value adds into the slots without a LayerNorm would let an injected signal scale linearly. If every path is normalised, I'll inject the deviation further downstream.

```python
for i in range(28,60): print(i, A[i][:220])
```