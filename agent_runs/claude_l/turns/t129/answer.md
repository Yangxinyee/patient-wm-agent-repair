The output changes the picture. Over hours 13–24 the model's insulin ×2 shift grows (−1.07 to −4.87), and so does my estimated true shift (−3.4 to −18.7, as the dose history builds up). Their ratio stays fairly constant at about 0.2–0.3: 0.31 at hour 13, then 0.20–0.26. So the model's response is roughly the right shape, just about 4.4 times too weak for insulin.

Norepinephrine is more awkward:
- ×2 gives only +0.62 against an estimated +4.9;
- ×1/2 gives −0.82 against −2.45.

That asymmetry means the model saturates once rates exceed the recorded maximum of 1.0.

This suggests a principled edit: make the model perceive dose innovations λ times larger, i.e. post-bundle state ≈ state(expected dose + λ·(dose − expected dose)). Teacher-forced predictions would then improve (the error is in the residual), and rollouts would scale up by about λ. Injecting along the model's own dose-response direction needs some replica pieces, so I'll first recover my probe-comparison code.

```python
import inspect
for f in [enc_bundle2, split_atoms, fit_sub]:
    try:
        s=inspect.getsource(f); print('#####',f.__name__,len(s)); print(s[:1500])
    except Exception as ex: print(f.__name__,'no source',ex)
```