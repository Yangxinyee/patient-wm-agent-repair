The pilot’s transport differs from its baseline mainly in the two dose-decoding rows and their output columns. That makes the decoder replacement reversible, but I still need to verify the factual compensation and no-drug gate. A nonzero signal when no drug is present would be evidence of a source of drift.

```python
import dis
print("=== RECOVER HELPER DEFINITIONS ===")
for _target202 in ["latent187", "prelude155", "prelude_parts145"]:
    _found202 = False
    for _key202 in list(globals()):
        _obj202 = globals()[_key202]
        if isinstance(_obj202, str) and ("def " + _target202 + "(") in _obj202:
            _start202 = _obj202.index("def " + _target202 + "(")
            _end202 = _obj202.find("\ndef ", _start202 + 5)
            if _end202 < 0: _end202 = min(len(_obj202), _start202 + 7000)
            print(_key202, _obj202[_start202:_end202][:7000])
            _found202 = True
            break
    if not _found202:
        print("\nBYTECODE:", _target202)
        dis.dis(globals()[_target202])
print("\n=== CURRENT FACTUAL SIGNALS ===")
print("decoder full mean/std:", signal192.mean(0), signal192.std(0))
_sig202 = np.asarray(xlatent199) @ restricted199
print("decoder restricted mean/std:", _sig202.mean(0), _sig202.std(0))
print("carrier payload mean/std:", payload154.mean(0), payload154.std(0))
print("reserved KV rows/output columns:", rows199, cols199)
print("candidate compensation shapes:", dw192.shape, db192.shape)
print("compensation applied:",
      np.max(np.abs(np.asarray(pilot192["core.prelude.0.ffn.2.weight"]) -
                    np.asarray(orig192["core.prelude.0.ffn.2.weight"]))),
      np.max(np.abs(np.asarray(pilot192["core.prelude.0.ffn.2.bias"]) -
                    np.asarray(orig192["core.prelude.0.ffn.2.bias"]))))
```
