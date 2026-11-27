Writer v2 is effectively inert. The edited model's predicted means move by about 0.007 mg/dL (MAE) and 0.009 mmHg on average. The lag-1 correction it was designed to add (≈ −2 per unit of insulin innovation, ≈ +4 for norepinephrine) would move them by roughly 0.5 or more. Either the gated units don't fire in the real model the way my reconstruction says, or z writes at drug bundles barely reach the state. A states pass (original vs edited) on the same 99 stays will tell these apart: compare the measured z-posterior change at drug bundles with my designed `dzD`. If measured dz ≈ designed dz, the bottleneck is propagation (z→state). If dz ≈ 0, the units aren't firing. First I need the stay IDs.

```python
sel=json.load(open('/work/sel_states.json'))
sel=sel if isinstance(sel,list) else sel.get('stays',sel)
print(len(sel)); print(json.dumps([int(s) for s in sel]))
```