Function sources can't be recovered, but the functions and their config dicts are still in the kernel and callable. ARCHITECTURE.md is short (5.2k characters), so I will re-read it in full to rebuild the model in my head. I will also print the config dicts (`MEAS_CFG`, `START_CFG`, `END_CFG`), the short `notes` string, and the names each key function calls. Then I can work out how `encode_bundles` produced the near-exact match.

```python
print(open('/model/ARCHITECTURE.md').read())
print("notes var:", notes)
print("MEAS_CFG", MEAS_CFG); print("START_CFG", START_CFG); print("END_CFG", END_CFG)
for k in ['encode_bundles','atom_vectors','deepsets']:
    co = globals()[k].__code__
    print(k, 'args', co.co_varnames[:co.co_argcount], 'defaults', globals()[k].__defaults__, '\n  names:', co.co_names)
```