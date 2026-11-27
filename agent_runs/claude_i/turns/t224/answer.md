The architecture file is short (5.2k characters), and the kernel still has many helpers: core_step, head_value, pibar, zpost2, the deconvolution tools and others. To design a dynamics edit I need the exact definitions of the state update, the heads and the dose-edit readout, so I will print those sections (from line 30 on). This is reading, not a test. What I want to learn is whether each bundle writes into h additively with a per-dimension decay, which would allow a dedicated decaying dose-memory channel. I also want to see how the value head turns the state into a distribution, which decides whether a mean shift can be added cleanly.

```python
L = A.split('\n')
print('\n'.join(L[30:]))
```