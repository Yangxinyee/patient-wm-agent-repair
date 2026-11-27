`tu` holds the tuning events (27,609 rows), and `meas_parts` returns a 192-d atom vector. So I can build approximate e_k for reading bundles with `enc_e`. I'll print the event table's columns and two rows, and check whether `UB` is the constant no-action u_k vector. Then I can compute the prelude attention at non-drug bundles.

```python
print(list(tu.columns)); print(tu.head(2).to_string()[:400])
print('UB norm', float(UB.norm()), 'vs U_const?', [k for k in globals() if k.lower().startswith('u0') or k.lower().startswith('ucon')])
```