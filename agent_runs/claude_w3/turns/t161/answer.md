vocab.json gives per-code medians and IQRs, including the drug rate and dose scales (drug A rate: median 1.5, IQR 1.0; drug B rate: median 0.46, IQR 0.6). Next I'll check the bundle DataFrame columns and the signature of `set_enc_`. Then I can rebuild e and u the way `zpost_err` did (`atom_vec` with the `P0` parameters, then `set_enc_`) and plug them into my fresh z check. If that drops the z error from 0.59 towards 0.10 or lower, the old atom-level encoder is the more accurate one and I'll build on it.

```python
print(inspect.signature(set_enc_), list(B_[0].columns), len(B_[0]))
print(B_[1].iloc[0].to_dict() if len(str(B_[1].iloc[0].to_dict())) < 400 else '')
print(type(P0), len(P0), list(P0.keys())[:5])
```