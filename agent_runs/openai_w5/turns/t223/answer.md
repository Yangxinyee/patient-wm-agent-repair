The private channel looks close to linear, but that alone does not establish a linear response at the readout. The exposure proxy may also move when a dose changes. I’ll measure the factual centering residuals and recover the reading-to-state mapping before increasing any strengths. If the proxy is dose-invariant, its cancellation cannot explain the asymmetry.

```python
import numpy as _r197_np

_r197_specs = _r196_specs
_r197_private = _r197_np.array([61, 21, 15, 66])
_r197_X = _r147_X
_r197_events = _r148_event_array

print("=== FACTUAL EXPOSURE DECODING AND CENTERING ===")
for _r197_sp in _r197_specs:
    _r197_mask = _r197_events == _r197_sp["event"]
    _r197_proxy = _r197_sp["proxy"].predict(_r197_X)
    _r197_delta_row = (
        _r197_np.asarray(_r197_sp["row_signal"])
        - _r197_np.asarray(_r197_sp["row_control"])
    )
    _r197_delta_bias = float(
        _r197_sp["bias_signal"] - _r197_sp["bias_control"]
    )
    _r197_decoder_row = (
        _r197_delta_row / _r197_sp["c"]
        + _r197_sp["proxy"].coef_
    )
    _r197_decoder_bias = (
        _r197_delta_bias / _r197_sp["c"]
        + float(_r197_sp["proxy"].intercept_)
    )
    _r197_decoded = _r197_X @ _r197_decoder_row + _r197_decoder_bias
    _r197_true = _r147_dose[:, _r197_sp["drug"]]
    _r197_centered = _r197_decoded - _r197_proxy
    print(_r197_sp["event"], {
        "n": int(_r197_mask.sum()),
        "true_mean": float(_r197_true[_r197_mask].mean()),
        "decoded_RMSE": float(_r197_np.sqrt(
            _r197_np.mean((_r197_decoded[_r197_mask] -
                          _r197_true[_r197_mask]) ** 2))),
        "proxy_RMSE": float(_r197_np.sqrt(
            _r197_np.mean((_r197_proxy[_r197_mask] -
                          _r197_true[_r197_mask]) ** 2))),
        "centered_sd": float(_r197_centered[_r197_mask].std()),
        "proxy_coef_norm": float(_r197_np.linalg.norm(
            _r197_sp["proxy"].coef_)),
        "proxy_private_coefficients":
            _r197_sp["proxy"].coef_[_r197_private].tolist(),
        "decoder_private_coefficients":
            _r197_decoder_row[_r197_private].tolist(),
        "decoder_ordinary_norm": float(_r197_np.linalg.norm(
            _r197_np.delete(_r197_decoder_row, _r197_private)))
    })

print("\n=== READING-CACHE IMPLEMENTATION REFERENCES ===")
print("reading_cache globals:", reading_cache.__code__.co_names)
print("reading_cache string constants:",
      [v for v in reading_cache.__code__.co_consts if isinstance(v, str)])
print("reconstruct_head string constants:",
      [v for v in reconstruct_head.__code__.co_consts if isinstance(v, str)])

print("\n=== EXISTING READING CACHES ===")
_r197_caches = []
for _r197_n, _r197_o in list(globals().items()):
    if isinstance(_r197_o, dict) and any(
        isinstance(v, _r195_pd.DataFrame) and len(v) == 4750
        for v in _r197_o.values()
    ):
        _r197_caches.append((_r197_n, _r197_o))
for _r197_n, _r197_o in _r197_caches[-12:]:
    print(_r197_n, {
        k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        for k, v in _r197_o.items()
    })

print("\n=== POSTERIOR HELPER REFERENCES ===")
print("posterior_features4 globals:", posterior_features4.__code__.co_names)
print("context_from_saved globals:", context_from_saved.__code__.co_names)
```