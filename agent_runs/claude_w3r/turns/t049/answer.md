Progress: with exact GELU and gap input log(dt in seconds)/10, the z posterior error falls from about 0.4 to about 0.04–0.06, against a typical |z| of 0.58. So the activation and the gap scaling are now right, and the earlier mismatch came mostly from the gap input. The remaining error is likely from the unit-index mapping or from whether END atoms carry unit, rate and dose.

Next I search all 24 assignments of the four units to rows 2–5, both rate_unit orders and both END-atom options. A combination that drives the error to roughly 1e-3 would confirm the input side is exact. If the error stays near 0.04 for every combination, something else is still missing.

```python
units4 = ['mmHg','mg/dL','units/hour','mcg/kg/min']
code_unit = {'map_cuff':'mmHg','map_arterial':'mmHg','glucose_finger':'mg/dL','glucose_lab':'mg/dL','insulin':'units/hour','norepinephrine':'mcg/kg/min'}
res_u=[]
for perm in itertools.permutations([2,3,4,5]):
    um=dict(zip(units4,perm))
    for ruo in [(2,3),(3,2)]:
        for ea in [True, False]:
            o=dict(unit={c:um[u] for c,u in code_unit.items()}, ru={'insulin':ruo[0],'norepinephrine':ruo[1]}, end_attrs=ea)
            r_=zerr(o,'gelu','l10'); res_u.append((r_['A']+r_['O'], perm, ruo, ea, r_))
res_u.sort(key=lambda t:t[0])
for t in res_u[:8]: print(t)
```