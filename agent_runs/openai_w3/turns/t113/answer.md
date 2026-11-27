The existing extractors substantially understate dose changes, so using them unscaled would repeat the earlier ineffective repair. Also, the factor blocks have different lengths; I’ll correct that summary and compare the direct attention features. I’m looking for a reliably scalable dose signal with low factual variation before choosing an insertion point.

```python
import numpy as np
import torch

print("CORRECT HELD-OUT FACTOR RESPONSES")
for key in [(1.0, 1e-8), (10.0, 1e-8)]:
    pred = DX88 @ contrast_fits88[key]
    start = 0
    print("fit", key)
    for drug, factor, indices in contrast_meta88:
        stop = start + 4*len(indices)
        mask = ~DT88[start:stop]
        target = DY88[start:stop][mask].mean(axis=0)
        response = pred[start:stop][mask].mean(axis=0)
        print(drug, factor, "target", np.round(target, 3),
              "response", np.round(response, 3))
        start = stop
    assert start == len(DX88)

print("\nATTENTION-POOL FIT VALIDATION")
for head in range(4):
    for lam in [0.1, 1.0, 10.0]:
        b, centre = pool_fits89[(head, lam)]
        pred = PD89[:, head, :] @ b
        fs = (PF89[:, head, :] - centre) @ b
        mask = ~DT88
        slopes = [np.sum(pred[mask, j]*DY88[mask, j]) /
                  max(np.sum(DY88[mask, j]**2), 1e-12) for j in range(2)]
        print(head, lam, "slopes", np.round(slopes, 3),
              "factual SD", np.round(fs.std(axis=0), 3))

print("\nREADOUT WORKING VARIABLES")
for k, v in sorted(list(globals().items())):
    if k.endswith("97"):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape), "sample",
                  np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v).reshape(-1)[:3])
        elif isinstance(v, (int, float, str, tuple)):
            print(k, repr(v)[:250])

print("\nAVAILABLE REPLAY INPUTS")
for k in ["h81", "h80", "h82", "flowed81", "prev81", "idx81",
          "u80", "u81", "z81", "dt81", "ctx81", "S", "W"]:
    if k in globals():
        v = globals()[k]
        print(k, {q: np.shape(x) for q, x in v.items()} if isinstance(v, dict) and k != "W"
              else (len(v) if k == "W" else np.shape(v)))

print("\nREADOUT-RELATED SAVED SOURCE")
for k, v in sorted(list(globals().items())):
    if isinstance(v, str) and len(v) > 500 and "last_numeric" in v:
        pos = v.find("last_numeric")
        print(k, "length", len(v), v[max(0, pos-350):pos+1100])

print("\nNUMERIC VOCAB EXAMPLE", VOCAB["numeric"]["glucose_finger"])
print("\nCURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs 101-102; forwards 7. Existing prelude FFN-hidden dose extractors "
            "underrespond (held-out slopes about 0.23-0.43); factual score SD decreases "
            "with covariance penalty. Correcting unequal factor-block summaries and "
            "comparing attention-pooled extraction before calibration. No edits.\n")
```
