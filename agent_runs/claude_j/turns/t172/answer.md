The kernel has the replica pieces: `step(out_prev, h_prev, dt_s, e_k, u_k, z_k, Mem, ages)`, `encode`, and atom builders. It also has several drug-atom variants (`drug_atom2`, `drug_end` with "end_attrs/end_kq/end_ekind" options) and z-error checks.

So in the undocumented runs I was evidently still settling how drug atoms are encoded, especially the infusion-end atoms, and checking the z posterior. That matters: counterfactual states, and any clean dose channel, both depend on encoding drug atoms exactly.

I will reread the "Input" section of ARCHITECTURE.md, lines 7–29, about 1.5k chars, to pin down the atom construction. Then I will test the replica on drug bundles against the tool's states.

```python
print('\n'.join(L_[7:30])[:2600])
```