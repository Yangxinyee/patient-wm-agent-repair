`code_id` has no `'norepinephrine'` key. It may cover only some codes or use different names. To stop depending on an unverifiable helper, I will print `code_id` and the `embed.*` tensor names. Then I build the NE atom directly from the formula verified in run 236:
- **START atom:** code[9] + kind[0] + ekind[2] + rate/dose bins + numeric([0, 0, z_dose, 1, z_rate, 1]) + bias + unit[4] + rate_unit[3].
- **END atom:** kind[2], with no rate or dose.

```python
print(code_id)
print({n:SHAPES[n] for n in NAMES if n.startswith('embed.') and not n.startswith('embed_transition')})
```