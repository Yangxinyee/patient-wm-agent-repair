The function defaults recovered the encoding choices I had settled on:
- unit-table rows: MAP 2, glucose 3, insulin 4, norepinephrine 5;
- rate-unit rows: insulin 2, norepinephrine 3;
- `use_kq=True`;
- left-sided bins;
- GELU activations;
- END atoms routed into the measurement encoder ('e_route'), which gave the lowest z-posterior error in run 93.

The atom table ATM holds one row per event with code, kind, event kind, value, rate, dose and bundle index. To write a clean, file-backed implementation I still need the code ids, the per-code vocab entries and the numeric normalisation. Next I print VOCAB['codes'], code_id/CODEN and NUMV compactly. If the vocab lists units or categorical ids per code, I can confirm the unit-row mapping independently rather than rely on the fit alone.

```python
import json
print(code_id); print(CODEN)
vc=VOCAB['codes']
print(type(vc), (list(vc.items())[:3] if isinstance(vc,dict) else vc[:3]))
print(json.dumps(VOCAB['numeric'])[:1500])
print(list(NUMV.items())[:2])
```