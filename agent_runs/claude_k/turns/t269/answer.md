The model sizes are in `vocab.json["config"]`. The architecture text also shows that set encoders end in a LayerNorm, so u_k has a fixed scale and dose changes move it only slightly, which explains the model's weak dose sensitivity. In one run I'll read the head count from the config and compute, at drug bundles, the attention the four slot queries give each token (4 slots, e_k, u_k, z). I'll use original states, `u_rec` from the recorded rates (taken from `st_s` if its `hi`/`hn` are rates) and `e_empty()` for e_k. If attention to z is ≥ 0.1, amplifying a dedicated z channel through `z_in` is viable.

```python
cfg=VOCAB['config']; print({k:v for k,v in cfg.items() if not isinstance(v,(list,dict))})
print(st_s[st_s.hi>0].head(2).round(3).to_string())
```