Run 82 lists 42 embedding and encoder tensors. Two things stand out. There is an `embed.kq` table the architecture text does not mention. Each set encoder is an MLP 192→768→192, applied before and after the sum over atoms, with a LayerNorm-style affine at the end. Next I'll list the `core.*` tensors, which hold the recurrent step I most need to reproduce exactly.

```python
show(['core'])
```