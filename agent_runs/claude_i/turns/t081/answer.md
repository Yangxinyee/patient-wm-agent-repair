The kernel functions exist but their sources can't be shown, since they were defined in executed code. Before deciding whether to rebuild the full model offline, I need to see how big the core is. Next I'll print the core.* and encode_action.* tensor names with shapes, compactly, plus the call signatures of my old helpers (bundle_eu, set_encode, zpost_check, head_value2, hidden_of). If the core has only a few standard layers whose shapes fit the documented structure, a re-implementation checked against fwd_3 states is worth doing. If it is large or ambiguous, I'll stick to edits tuned with the forward tool.

```python
core = [(n, tuple(SHAPES[n])) for n in NAMES if n.startswith('core.')]
print(len(core))
print('; '.join(f"{n[5:]}{list(s)}" for n, s in core)[:3000])
print([ (n, tuple(SHAPES[n])) for n in NAMES if n.startswith('encode_action.')])
for k in ['bundle_eu','set_encode','zpost_check','head_value2','hidden_of','head_inputs','atom_vec','run_var']:
    print(k, inspect.signature(globals()[k]))
```