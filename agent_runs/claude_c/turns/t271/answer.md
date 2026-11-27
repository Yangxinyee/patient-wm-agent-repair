The norepinephrine bands confirm the model has learned almost no norepinephrine effect. Only the 0.5-1 band's x1/2 response has the right sign: -1.05 in the original, -1.70 under head0 x4. N5 flattens it to about 0, and nothing cheap fixes this.

One lever is still untested. The b_in injection is only one part of the state update (the blocks' attention and FFN make up the rest). So amplifying b_in's output along the insulin dose-imprint direction might be more selective than the 0.17 SNR suggests. That SNR was measured against the whole reading-step variation, not b_in's share of it. I'll check the shapes of the core update tensors and look for saved dose-direction arrays in the kernel.

```python
print({n:SHAPES[n] for n in NAMES if n.startswith('core.b_in') or n.startswith('core.c_out') or 'delta_in' in n or n.startswith('core.a_in') or 'prelude_norm' in n})
arrs={k:v.shape for k,v in globals().items() if isinstance(v,np.ndarray) and v.ndim<=2 and (192 in v.shape or 768 in v.shape)}
print(dict(list(arrs.items())[:30]))
```