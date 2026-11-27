The architecture explains the NE x2 problem. A drug atom's encoding adds bin embeddings (embed.bins.rate and embed.bins.dose) to a numeric term that is linear in the z-scored rate and dose. The bins come from per-code edges, but the bin tables are shared with insulin. So NE rates above 1.0 land in rate bins 13-16, which NE never used in training (only insulin did), while the numeric part extrapolates linearly. The NE x2 rollout therefore feeds the action encoder an input combination it never saw.

Editing those shared bin rows would also change insulin's encoding and the factual fit, so that is not free. Next I'll read the middle of the architecture (set encoders, action encoder, state update) to see how u_k enters the state.

```python
print('\n'.join(l[:170] for l in lines[24:60])[:5200])
```