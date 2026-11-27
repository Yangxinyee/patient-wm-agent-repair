Each of obj, key_k and bidx holds, per stay, the row indices into SX plus a second element. The states' hour field gives exact bundle times (0.0833 h = 5 minutes after the hour). So I can line up states with bundles once I rebuild each stay's bundles with my build_bundles function.

Next I will print the build_bundles source and the second element of obj, to see the atom format my encoder expects and how events are passed in. I can then compute e_k and u_k for every SX row.

- **Confirms:** build_bundles yields one bundle per SX row, with matching times and atom counts.
- **Refutes:** they don't match. Then the alignment logic needs fixing before any core-step check.

```python
print('\n'.join(lines[10:34]))
print('obj[s0][1]:', str(obj[s0][1])[:600])
print('key_k[s0][1]:', str(key_k[s0][1])[:300])
print('bidx[s0][1]:', str(bidx[s0][1])[:300])
```