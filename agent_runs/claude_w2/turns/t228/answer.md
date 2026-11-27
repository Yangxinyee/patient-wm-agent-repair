Decoding from the head alone will not work.
- **Insulin, lag 1:** the best stacked decoder reaches a causal slope of only 0.48 (μ=10). Its factual error sd is 1.25, larger than the signal's sd of 1.09. Pushing μ higher raises the slope only to 0.6, while the error explodes to 2.9–7.1. A channel built on it would need roughly twice the gain to compensate, injecting about 5 mg/dL of noise and breaking the 0.005 NLL budget.
- **Norepinephrine:** better, but the slope is still only 0.64.

The residual regression also confirmed the lag-1-only form. The deviation coefficients are −1.26, −1.36, −0.64, and the net policy-level coefficients γ are −0.57, −0.21, −0.38, +0.16, +0.65. d1 regressed on the policy lags is ≈ p1 (β = 0.995).

So the dose amount must be made more visible upstream: the core's output state must carry a clean, roughly linear lag-1 dose signal for the head to read. Run 212 shows I had started this, as a rank-1 write from a prelude attention head into the residual stream. To choose the injection point I need the exact core-step code (where u enters, the b_in / blocks / coda path to 'out') and the encoder and core tensor names. That tells me whether actions have their own encoder weights, which I could edit without touching the measurement encoding. I am printing core_test.py and those names now.

```python
print(open('/work/core_test.py').read())
print([k for k in NAMES if not k.startswith('attr_heads') and not k.startswith('core.blocks') and not k.startswith('core.coda') and not k.startswith('core.prelude')])
print(sorted(set('.'.join(k.split('.')[:3]) for k in NAMES if k.startswith('core.'))))
```