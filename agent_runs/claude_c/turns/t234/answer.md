The vocabulary explains the NE x2 problem. NE rates are capped at 1.0, which lands in rate bin 12. Any doubled rate above 0.5 falls into bin 16, a bin NE never reached in training (same for dose bins 13-16), with z-scores well past the training maximum. So about half of the NE x2 edits feed the model unseen inputs, which explains its sign-wrong response.

The tensor shapes show that `encode_action` is a sum of mlp1 outputs over atoms, followed by mlp2 and a LayerNorm. `z_post` is a 2-layer MLP. Next I will read the Heads section. I want to know whether predictions use `z_head(z)` of the previous bundle's posterior. If they do, the posterior z (whose dim 26 tracks dose at drug bundles) gives a direct, selective route to the next prediction.

```python
i=A.find('## Heads'); print(A[i:i+2300])
```