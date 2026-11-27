Run 221 lists components beyond those in my recent notes: pending_mlp, filter_gate, last_numeric, gap, query_norm, embed_transition, and the core state-space tensors (a_in, b_in, c_out, delta_in, flow, coda).

Some of these could be a second route from dose to predictions that bypasses the prelude attention. One candidate is a "pending action" summary fed to the prediction query. If such a route exists, and its dose input is not renormalised the way the action token is, it could be amplified without the saturation that limits N5.

Earlier tests found that u=0 kills every drug effect. That fits u feeding such a pathway as well as the prelude, so the test does not rule this out.

Before re-reading the architecture I'll print the start of my notebook. It should hold my own architecture summary and is cheaper to read than the full document.

```python
t=open('/work/notes.md').read(); print(len(t)); print(t[:3200])
```