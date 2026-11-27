The README confirms the columns. My rebuilt bundles hold drug atoms with rate and dose; END atoms appear in later bundles.

Next I'll start the local re-implementation. First I need the full metadata of each code (unit, event kind and any index fields), to map atoms to the categorical embedding tables. I'll print it compactly.

```python
for c_ in VOCAB['codes']:
    print(c_)
```