The kernel still has my replica pieces: `encode`, `meas_atom`, `drug_atom2`, `visit_atom`, `bundle_vecs` and several `z_err*` checks. It also has `E_`, a table of 1,973 per-bundle errors on 100 stays. So the last run was probably checking the replica's z-posterior against the GPU `states` output, bundle by bundle: almost all bundles match to 1e-6 and a few are off by up to 0.1. The architecture printout was cut off at the heads section. Next I'll read the rest of it and look at which bundles hold the large errors, to see whether they share one cause (for example drug END atoms).

```python
A = open('/model/ARCHITECTURE.md').read()
i = A.find('**other heads:**')
print(A[i:])
print(E_.head(10)); print(E_.columns.tolist())
print(E_.describe())
```