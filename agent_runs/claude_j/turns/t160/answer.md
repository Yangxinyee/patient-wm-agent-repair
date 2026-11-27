`CFG` holds per-drug tuples (probably unit-table indices), and `cfg` adds an `end_attrs` switch (whether infusion-end atoms carry rate and dose). To see which choices `bundle_vecs` makes for drug atoms, I'll print a compact per-line trace of the names it uses. That shows where `CFG`, `drug_end` and the end-atom handling come in, so I can test the alternatives.

```python
lines={}
for x_ in dis.get_instructions(bundle_vecs):
    if x_.opname in ('LOAD_GLOBAL','LOAD_ATTR','LOAD_METHOD','LOAD_CONST','LOAD_FAST','STORE_FAST','COMPARE_OP','CONTAINS_OP') and x_.argval is not None:
        lines.setdefault(x_.positions.lineno,[]).append(str(x_.argval)[:12])
for l_ in sorted(lines): print(l_, ' '.join(lines[l_])[:120])
```