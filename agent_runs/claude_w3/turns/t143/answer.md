I'll keep each reply brief and stick to the model internals. I'll refer to the readings as G (glucose_finger) and P (map_arterial), and to the two drugs as drug A and drug B.

Next I need the recurrent block's sub-layer names (to see the cross-attention and age wiring) and the layout of the saved state file. Then I can write a clean core step and compare it bundle by bundle against the saved states. A low error on the first bundles would confirm the wiring; a high one tells me which component to fix.

```python
bl0 = [n.replace('core.blocks.0.0.','') for n in NAMES if n.startswith('core.blocks.0.0.')]
print(bl0)
print(F4.files, [F4[k].shape for k in F4.files][:8])
```