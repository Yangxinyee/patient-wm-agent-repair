`stay_bundles` gives a list of per-bundle DataFrames (54 for this stay), and `enc_bundle` returns (e, u, n1, n2). Next I'll check that the bundle count matches the tool's states for this stay, and look at one action bundle's columns, so I can line up local encodings with tool states row by row.

```python
sel=S4['stay_id']==s0; print(sel.sum(), S4['hour'][sel][:6], S4['has_insulin'][sel][:6])
print(B[1].columns.tolist()); 
ia=[i for i,bb in enumerate(B) if bb['code'].astype(str).str.contains('insulin').any()][:2] if 'code' in B[1] else []
print(ia, (B[ia[0]].to_string()[:600] if ia else ''))
```