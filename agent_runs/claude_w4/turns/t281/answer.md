fwd_6 tested the v3 head-1 carrier at gain 1 for both drugs. The truth on these stays is about −13 for insulin x2 and +5 for NE x2. Against that, v3 made the dose responses weaker than the original model's:
- insulin x2: −0.145 (original −0.348)
- insulin x1/2: +0.066
- NE x2: +0.010
- NE x1/2: −0.163

The injected slot vector is either washed out downstream or acts with the wrong sign. The first-hour shift at one stay went from −0.10 to +0.99, which points to the wrong sign. fwd_7 holds original and edited states on the same factual stays, so I can see directly how the injection spreads.

Runs 205–251 left no notes and the replica code is gone. Before analysing fwd_7 I'll save this result to the notebook and re-read the architecture, so I can tell how slot changes reach the glucose and MAP heads. If the architecture shows a normalisation or gating step between the injected slots and the heads, that would explain the washout and guide where to inject.

```python
note = """
## Restart 8 (runs 252-)
- Runs 205-251 were NOT noted (lost); run 251 crashed (memory). Current edit = v3(gI=1,gN=1,sP=0) == /work/v3_g1_1_0.npz (kv rows 240:288, wo cols 48:96).
- fwd_6 (dose_edit both, tuning sel100, v3(1,1,0)): EDITED ins x2 -0.145, x1/2 +0.066, NE x2 +0.010, NE x1/2 -0.163
  vs ORIGINAL -0.348/+0.208/+0.018/-0.245. => v3 at gain 1 slightly WEAKENS dose response (truth -12.96/+6.48/+4.94/-2.47).
  stay 2020000064 h13 ins x2 shift edited +0.99 vs orig -0.10 (sign wrong?).
- fwd_7: states both (edited=v3(1,1,0)) tuning sel100, 5218 bundles: use to see how injection propagates.
"""
open('/work/notes.md','a').write(note)
sel = json.load(open('/work/sel100_both.json')); print(type(sel), (list(sel.keys()) if isinstance(sel,dict) else len(sel)))
print(open('/model/ARCHITECTURE.md').read())
```