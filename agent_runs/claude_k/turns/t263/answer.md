The edit fires as intended: z at drug bundles moves by 0.15 in norm, against a natural per-dimension spread of 0.21, and nothing moves elsewhere. But the state barely responds: `out` changes by 0.0002 (natural spread 0.26) and the h slots by 0.009 (natural spread about 10.6). So z is a weak channel into the state, and no gain I could safely use would reach the required effect. I need a stronger route from the dose to the state. Candidates are the prelude and the state-update input. I'll re-read only the relevant architecture lines (prelude, state update, slots, how z and u enter) to find a high-gain entry point.

```python
txt=open('/model/ARCHITECTURE.md').read()
L=txt.splitlines(); print(len(L), len(txt))
keys=('prelude','update','slot','u_k','z_t','posterior','gate','write','decay','out =','out_','h_')
sel_l=[(i,l) for i,l in enumerate(L) if any(k in l for k in keys)]
for i,l in sel_l[:45]: print(i, l[:160])
```