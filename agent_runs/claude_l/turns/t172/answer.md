No age definition or register layout gives an exact match. The best constant-shift fit leaves a 1.5 residual, so I can't tell whether the error is in the register features or is noise from the tool's 4 prior draws of z. To remove that noise I'll spend one forward pass on a probe. I'll zero `z_head.weight` so that z has no effect, and run a factual pass on the same 100 batch1 stays. The tool's predicted means then become deterministic functions of the state, which I have exactly, so my head replica can be checked and fitted against a noise-free target. In this run I save the probe edit and print the stay ids.

```python
b1 = json.load(open('/work/batches.json'))['batch1']; print(len(b1))
print(json.dumps([int(x) for x in b1]))
save_edit('z_head.weight', np.zeros(SHAPES['z_head.weight'], dtype=np.float32))
print(edited())
```