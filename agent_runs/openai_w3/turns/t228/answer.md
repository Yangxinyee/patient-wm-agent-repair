Attention does not block the no-action signal: it reaches the state update almost unchanged. The restricted decoder does suppress each drug’s signal when only the other drug is recorded, so I’ll inspect the posterior gate neurons next. If disabling those gates removes the no-action signal, an explicit no-action gate may fix this without disturbing drug bundles.

```python
print("=== POSTERIOR GATE DIAGNOSTIC ===")
print("posterior shapes:", {k: tuple(np.asarray(v).shape) for k,v in post187.items()})
print("gate rows / latent reserves:", rows154, reserved154)
_xzero205 = tensor153(np.concatenate([C154, E154, np.zeros_like(U154)], axis=1))
_xfact205 = tensor153(np.concatenate([C154, E154, U154], axis=1))
_hzero205 = F.relu(F.linear(_xzero205, tensor153(post187["z_post.0.weight"]),
                            tensor153(post187["z_post.0.bias"])))
_hfact205 = F.relu(F.linear(_xfact205, tensor153(post187["z_post.0.weight"]),
                            tensor153(post187["z_post.0.bias"])))
print("gate activations, zero action min/mean/max:")
print(np.stack([_hzero205[:,rows154].numpy().min(0),
                _hzero205[:,rows154].numpy().mean(0),
                _hzero205[:,rows154].numpy().max(0)]))
print("gate activations, factual min/mean/max:")
print(np.stack([_hfact205[:,rows154].numpy().min(0),
                _hfact205[:,rows154].numpy().mean(0),
                _hfact205[:,rows154].numpy().max(0)]))
_hdisabled205 = _hzero205.clone()
_hdisabled205[:,rows154] = 0
_zdisabled205 = F.linear(_hdisabled205, tensor153(post187["z_post.2.weight"]),
                         tensor153(post187["z_post.2.bias"]))[:,:32]
_rawdisabled205 = F.linear(_zdisabled205, tensor153(zin154), tensor153(zb154))
_ndisabled205 = arr183(layer_norm(_rawdisabled205, "core.prelude.0.norm1"))
print("disabled-gate restricted signal mean/RMS:",
      (_ndisabled205 @ restricted199).mean(0),
      np.sqrt(np.mean((_ndisabled205 @ restricted199)**2, axis=0)))
print("gate biases:", np.asarray(post187["z_post.0.bias"])[rows154])
print("gate output column norms:",
      np.linalg.norm(np.asarray(post187["z_post.2.weight"])[:,rows154],axis=0))

print("\n=== ACTION CONSTANT CHANNEL ===")
print([k for k in NAMES if k.startswith("encode_action") and "norm" in k])
for _name205 in [k for k in NAMES if k.startswith("encode_action") and k.endswith("norm.weight")]:
    _prefix205 = _name205[:-6]
    _g205 = np.asarray(load_original(_name205))
    _b205 = np.asarray(load_original(_prefix205+"bias"))
    if _g205.shape == (192,):
        _den205 = np.sum(_b205/_g205)
        _c205 = (1/_g205)/_den205
        _v205 = U154 @ _c205
        print(_name205, "denominator", _den205,
              "constant min/mean/max:", _v205.min(), _v205.mean(), _v205.max())
        action_constant205 = _c205.copy()
```
