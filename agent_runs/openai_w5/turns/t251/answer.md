The kernel still has the head-reconstruction helper and its intermediate features. The inventory stopped on a vocabulary-format error, before changing any weights. I’ll inspect the helper and the register vectors it produced; an incorrect feature layout would explain the loss mismatch, while a correct layout would shift attention to timing and latent sampling.

```python
print("=== CURRENT HEAD ROWS ===")
print(head_rows.columns.tolist())
print(head_rows.head(6).to_string(index=False))
print("\n=== REGISTER VECTORS ===")
print(head_ln[:8])
print("last bins", head_vb[:12], "previous-state indices", head_prev[:12])
print("\n=== PRIVATE CHANNEL AND NEURON CONFIG ===")
print("pairs", private_pairs, "private", priv)
print("reserved", reserved, "sensor neurons", sensor_neurons)
print("\n=== NUMERIC SPECIFICATIONS ===")
for key, val in num_spec.items():
    print(key, str(val)[:550])
print("\n=== HEAD HELPER SOURCE ===")
try:
    print(inspect.getsource(make_head_inputs212))
except Exception as exc:
    import dis
    print(type(exc).__name__, str(exc))
    print("locals", make_head_inputs212.__code__.co_varnames)
    print("constants", [repr(x)[:200] for x in make_head_inputs212.__code__.co_consts])
    dis.dis(make_head_inputs212)
print("\n=== FORWARD FILES ===")
for p in sorted(Path("/work").glob("fwd_*")):
    print(p.name, [q.name for q in p.iterdir()])
```
