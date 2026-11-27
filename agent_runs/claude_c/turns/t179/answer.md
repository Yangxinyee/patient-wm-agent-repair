Sharpening head 0's queries ×3 left every readout unchanged (−1.83 / +0.75 / −0.51 / −0.58). Head 0's attention logits must be close to uniform, so the action token gets a fixed share of attention and content-driven re-weighting can't raise it. That also suggests the latent z token is weak for another reason. Its input map may be dominated by a constant bias that LayerNorm then normalises away, which would explain why my ×5 scalings of z26 did nothing.

I'll check whether the latent-input tensors include a bias and what shapes the latent-related tensors have. If `z_in` has a bias, zeroing it could make the z token track z26, which at drug bundles is a clean dose readout.

```python
print([ (n, SHAPES[n]) for n in NAMES if 'z_' in n or 'z.' in n ])
```