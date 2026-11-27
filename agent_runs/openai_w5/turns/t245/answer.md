This repeats the previous diagnostic; no new weights have changed. The dose and centering terms are separable. I’ll now check spare-channel numerical noise and screen correction strengths locally. If stronger corrections exceed the factual-loss budget, centering must improve before amplification.

```python
import torch.nn.functional as F

print("=== PRIVATE COMMON-MODE SCALE ===")
for _p218 in private_pairs:
    for _n218 in ["head_q", "head_wc"]:
        _x218 = np.asarray(globals()[_n218])
        _cm218 = _x218[:, _p218].mean(axis=1)
        _dm218 = _x218[:, _p218[0]] - _x218[:, _p218[1]]
        print(_n218, _p218,
              "common quantiles", np.round(np.quantile(_cm218, [0,.1,.5,.9,1]), 6),
              "difference quantiles", np.round(np.quantile(_dm218, [0,.1,.5,.9,1]), 7))

print("=== VOCAB ===")
print("config", V["config"])
print("vocab keys", list(V))
print("numeric example", num_spec[("glucose_finger", "value")])

_labels218 = np.zeros(len(head_rows), dtype=np.int64)
for _ev218 in sensor_neurons:
    _ix218 = np.flatnonzero(head_rows.event.to_numpy() == _ev218)
    _labels218[_ix218] = np.clip(
        np.searchsorted(
            num_spec[(_ev218, "value")]["fine_bin_edges"],
            head_rows.value.to_numpy()[_ix218], side="left"
        ) + 1, 1, 16
    )

_x218 = torch.as_tensor(head_wc, dtype=torch.float32)
_h218 = F.gelu(F.linear(
    _x218, BASE217["attr_heads.value_bin.0.weight"],
    BASE217["attr_heads.value_bin.0.bias"]
))
_tname218 = next(n for n in NAMES
                 if n.startswith("embed_transition.value_bin")
                 and n.endswith("weight"))
_ci218 = np.array([code_id[e] for e in head_rows.event], dtype=np.int64)
_tr218 = BASE217[_tname218][torch.from_numpy(_ci218 * 17 + head_vb.astype(np.int64))]
_y218 = torch.from_numpy(_labels218)

print("=== LOCAL FACTUAL STRENGTH SCREEN; NO EDITS ===")
_losses218 = {}
for _s218 in [0., 1., 1.5, 2., 3.]:
    _ow218 = BASE217["attr_heads.value_bin.2.weight"].clone()
    _ow218[:, reserved] *= _s218
    _logits218 = F.linear(
        _h218, _ow218, BASE217["attr_heads.value_bin.2.bias"]
    ) + _tr218
    _loss218 = F.cross_entropy(_logits218, _y218, reduction="none").numpy()
    _losses218[_s218] = _loss218
    print("strength", _s218, "NLL", round(float(_loss218.mean()), 6),
          "by sensor", {
              e: round(float(_loss218[head_audit[e]["ix"]].mean()), 6)
              for e in sensor_neurons
          })
print("Increment over current:", {
    s: round(float((v - _losses218[1.]).mean()), 6)
    for s, v in _losses218.items()
})
```
