Hardly any of the dose surprise can be read from the state (cross-validated R² at most 0.075). The model keeps the drug history mostly as "what the readings imply was given", not as the actual dose. A small edit to the heads alone therefore cannot make predictions respond to dose changes. The dose signal has to be carried through the recurrent core.

To design and test such an edit offline, without spending forward passes, I need a complete CPU replica of one core step and of the value head, checked against the GPU states. Next I'll collect the configuration (number of heads, sizes) and the signatures of the replica pieces I already have.

```python
print(VOCAB['config'])
print(CFG, cfg)
for fnm in ['meas_atom','meas_atom2','meas_atom_gl','encode','encode_v','drug_atom','drug_atom2','drug_end','visit_atom','bundle_vecs','ctx_of','lin','mlp','ln','zsc','fbin','build_bundles']:
    f_ = globals().get(fnm)
    if f_ is not None:
        try: print(fnm, inspect.signature(f_))
        except Exception as ex_: print(fnm, ex_)
print(ACT, UNIT, NUM, code_id)
print(list(BUN.items())[0][1][:3] if isinstance(list(BUN.values())[0], list) else type(list(BUN.values())[0]))
```