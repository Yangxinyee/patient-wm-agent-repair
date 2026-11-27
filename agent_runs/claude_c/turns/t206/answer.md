The first chunk covers the input embedding, the set encoders and the core step. The action token u_k enters the prelude as one of 7 tokens and also enters z_post. The prelude's slot outputs go through `core.prelude_norm` to give E. E then drives the recurrent block. I'll read the remaining part, the heads and how values are predicted, to see where else the dose signal could be amplified.

```python
print(A_txt[3000:])
```