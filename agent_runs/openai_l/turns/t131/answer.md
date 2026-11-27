The retained kernels show a delayed insulin response and a faster-decaying norepinephrine response. That argues against fitting only the four reported numbers. I’ll first test whether a signal inserted just before the coda survives in the quiet state directions, and recover the counterfactual-bundle interface. Strong, isolated transmission would support this route; leakage would argue against it.

```python
import dis

print("=== BUNDLE AND COUNTERFACTUAL INTERFACES ===")
for n122 in ("bundle_inputs74", "make_cf81", "action_embedding67"):
    f122 = globals()[n122]
    print(n122, "dependencies:", f122.__code__.co_names)
    print("constants:", [x for x in f122.__code__.co_consts
                         if isinstance(x, (str, int, float))][:25])
    instructions122 = list(dis.get_instructions(f122))
    print("\n".join(f"{x.opname} {x.argrepr}" for x in instructions122[-32:]))

print("\n=== RETAINED DATA SCHEMAS ===")
for k122, v122 in list(globals().items()):
    if re.search(r"(60|63|67|74|81|96|102|103)$", k122):
        if isinstance(v122, (np.ndarray, torch.Tensor)):
            print(k122, tuple(v122.shape), str(v122.dtype))
        elif isinstance(v122, dict):
            print(k122, "keys:", list(v122)[:12])
        elif hasattr(v122, "columns"):
            print(k122, "shape:", v122.shape, "columns:", list(v122.columns)[:16])
        elif isinstance(v122, (list, tuple)):
            print(k122, type(v122).__name__, len(v122))

print("\n=== C O D A TRANSMISSION TEST ===")
idx122 = np.linspace(0, len(Hpre118)-1, 100).astype(int)
pre122 = torch.as_tensor(Hpre118[idx122], dtype=torch.float32)
cout122 = linear_backup118(pre122, "core.c_out")
post122 = layer63(cout122, "core.coda.0")
route122 = torch.as_tensor(R118, dtype=torch.float32)
transmit122 = []
leak122 = []
for j122 in range(4):
    pp122 = layer63(cout122 + route122[:, j122][None, None, :], "core.coda.0")
    pm122 = layer63(cout122 - route122[:, j122][None, None, :], "core.coda.0")
    d122 = ((pp122-pm122)/2).mean(1).numpy()
    transmit122.append(d122 @ R118)
    leak122.append(np.sqrt(np.mean(
        (d122 - (d122 @ R118) @ R118.T)**2)))
print("Private response matrix:",
      np.round(np.stack(transmit122).mean(1), 4))
print("Orthogonal leakage RMS:", np.round(leak122, 6))
print("Private direction Gram matrix:", np.round(R118.T @ R118, 4))
print("Current edits:", edited())
```