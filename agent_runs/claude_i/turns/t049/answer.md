The model underfits the insulin effect even on its own training stays. Its glucose residuals fall with the dose's deviation from what the policy would give for the reading (n = 1477):
- −1.59 (SE 0.49) per u/h for the dose just given (dev_t);
- −1.74 (SE 0.45) for the dose an hour earlier (dev_t−1).

This fits the rollout numbers: the model's one-step sensitivity is about −0.42 per u/h, against a true −2.0. So hypothesis (a) holds: the model is weak on dose everywhere, not only when extrapolating.

The right correction is a stronger response to (dose − E[dose | decision reading]). E[dose | reading] is a step function of the reading band: 0 below 150, 1 for 150–180, 2 for 180–220, 3 for 220–250, about 4 for 250–300, and 6 at 300 or more. Applied this way, the edit recovers the true structural dynamics even off-policy, and should slightly improve factual NLL rather than hurt it. A correction proportional to the raw dose would bias factual predictions by about 2–4 mg/dL and break the 0.005-nat limit.

Putting this correction where dose meets reading, with the right lags and per code, is hard to design blind. A local replica of the model would let me test edits on counterfactual doses offline. My last session had partly built one (z-posterior correlation 0.976). Next, I will list the functions already in the kernel to see what replica code exists. If an encoder/core implementation is there, I'll debug it against the stored states. If not, I'll decide between building one and a head-level correction tuned with forward passes.

```python
import types
fns = [k for k,v in globals().items() if callable(v) and not k.startswith('_') and isinstance(v, types.FunctionType)]
print(fns)
import inspect
for k in fns:
    try:
        src = inspect.getsource(globals()[k]); print("-----",k, len(src)); print(src[:600])
    except Exception as ex: print(k, "no source", ex)
```