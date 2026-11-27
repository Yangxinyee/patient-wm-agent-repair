The new derivative check produced a singular response matrix, so I won’t install an edit from it. This conflicts with the earlier nonzero sensitivities and needs explanation. I’ll inspect the helper and compare small and large perturbations; consistent nonzero responses would support using this pathway, while genuinely vanishing responses would refute the local calibration.

```python
import dis

print("=== DERIVATIVE DIAGNOSTICS ===")
print("J126 finite/range/norm:", np.isfinite(J126).all(),
      np.min(J126), np.max(J126), np.linalg.norm(J126))
print("A126:", A126)
print("J120 norm:", np.linalg.norm(J120))
print("read_hpre120 constants:", read_hpre120.__code__.co_consts)
dis.dis(read_hpre120)

print("\n=== EXPLICIT PERTURBATION CHECK ===")
d127 = int(np.argmax(np.linalg.norm(J120, axis=0)))
for eps127 in (0.25, 1.0, 4.0):
    pp127 = torch.zeros_like(hpre119)
    pp127[:, :, d127] = eps127 * delta119[d127]
    a127 = np.array(read_hpre120(hpre119 + pp127), copy=True)
    b127 = np.array(read_hpre120(hpre119 - pp127), copy=True)
    print("dimension/epsilon", d127, eps127,
          "output shape", a127.shape,
          "max difference", np.max(np.abs(a127-b127)),
          "derivative norm", np.linalg.norm((a127-b127)/(2*eps127)))

print("\n=== INDEX AND LOCAL HEAD DETAILS ===")
print("local_value_head signature:", inspect.signature(local_value_head))
print("local_value_head names:", local_value_head.__code__.co_names)
for name127 in ("posterior_rows_repair", "repair_start_rows101",
                "indices", "benchmark_stays"):
    obj127 = globals().get(name127)
    print(name127, describe115(obj127))
    if isinstance(obj127, (list, np.ndarray)):
        print("first:", obj127[:8])
```
