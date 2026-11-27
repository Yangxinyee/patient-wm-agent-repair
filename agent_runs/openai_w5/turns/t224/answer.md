The proxies ignore the private coordinates, but the decoding check used their training-feature matrix, which may have those coordinates cleared. Its large errors therefore are not valid transport evidence. I’ll repeat the check on actual head inputs and verify the reading cache. Low corrected decoding error would support the channel’s validity; persistently large error would challenge it.

```python
_r198_wc = _r149_wc32.detach().cpu().numpy().astype("float64")
_r198_Xfull = _r198_wc.mean(axis=0)

print("=== PRIVATE FEATURES: PROXY MATRIX VERSUS ACTUAL HEAD INPUT ===")
print("Proxy matrix private SD:", _r147_X[:, _r197_private].std(axis=0))
print("Actual head private SD:", _r198_Xfull[:, _r197_private].std(axis=0))
print("Ordinary-feature RMS difference:",
      float(np.sqrt(np.mean(
          np.delete(_r147_X - _r198_Xfull, _r197_private, axis=1) ** 2))))

_r198_decoder = {}
print("\n=== CORRECTED FACTUAL EXPOSURE CHECK ===")
for _r198_sp in _r197_specs:
    _r198_mask = _r148_event_array == _r198_sp["event"]
    _r198_dr = (
        np.asarray(_r198_sp["row_signal"]) -
        np.asarray(_r198_sp["row_control"])
    ) / _r198_sp["c"] + _r198_sp["proxy"].coef_
    _r198_db = float(
        (_r198_sp["bias_signal"] - _r198_sp["bias_control"]) /
        _r198_sp["c"] + _r198_sp["proxy"].intercept_
    )
    _r198_d = _r198_Xfull @ _r198_dr + _r198_db
    _r198_p = _r198_sp["proxy"].predict(_r198_Xfull)
    _r198_true = _r147_dose[:, _r198_sp["drug"]]
    _r198_decoder[_r198_sp["event"]] = (_r198_dr, _r198_db)
    print(_r198_sp["event"], {
        "decoded_mean": float(_r198_d[_r198_mask].mean()),
        "true_mean": float(_r198_true[_r198_mask].mean()),
        "decoded_RMSE": float(np.sqrt(np.mean(
            (_r198_d[_r198_mask] - _r198_true[_r198_mask]) ** 2))),
        "proxy_RMSE": float(np.sqrt(np.mean(
            (_r198_p[_r198_mask] - _r198_true[_r198_mask]) ** 2))),
        "centered_sd": float((_r198_d - _r198_p)[_r198_mask].std())
    })

print("\n=== READING CACHE AND MAPPING ===")
_r198_cache = _r140_cache
print("Cache types:", {k: type(v).__name__ for k, v in _r198_cache.items()})
print(_r198_cache["df"].head(6).to_string(index=False))
print("Previous state indices:", _r198_cache["previous"][:12])
print("Posterior helper references:", posterior_features4.__code__.co_names)
print("Context helper references:", context_from_saved.__code__.co_names)

print("\n=== SMALL HEAD RECONSTRUCTION ===")
_r198_head = reconstruct_head(
    _r198_cache, subset=np.arange(12), nsamples=4, keep_hidden=True
)
if isinstance(_r198_head, dict):
    print({
        k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        for k, v in _r198_head.items()
    })
elif isinstance(_r198_head, (list, tuple)):
    print([
        tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        for v in _r198_head
    ])
else:
    print(type(_r198_head).__name__,
          getattr(_r198_head, "shape", None))
```