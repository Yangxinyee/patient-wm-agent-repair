fwd_4 is the key result, and it is bad. With the edit, the tool's rollout gives:

| Readout | Edited model | Original model | Truth (g-estimate) |
|---|---|---|---|
| insulin x2 | +0.367 | −0.545 | ≈ −14.3 |
| insulin x1/2 | −0.244 | 0.003 | ≈ +7.15 |
| norepinephrine x2 | +0.054 | 0.193 | ≈ +4.11 |
| norepinephrine x1/2 | −0.008 | −0.173 | ≈ −2.05 |

So the edit has almost no effect in the tool's rollout, and the insulin signs are even wrong. This is far from what my local simulation predicted (about −1.3 mg/dL per u/h of residual dose on the next reading). fwd_5 was a states run of both models on 13 tuning stays, apparently to find where the tool and my local copy disagree. Runs 213–249 appear to have chased that disagreement ("SOLU3", bin-side variants), and none of it is in the notebook.

Next, I'll read the kernel's input history for those runs (short excerpts) to recover what I found, and check what SOLU3 is. Then I will write it into the notebook before doing anything else.

```python
H_ = In if 'In' in globals() else None
print('history len', len(H_))
n0 = len(H_) - 38
for i in range(n0, len(H_)-1):
    s = H_[i].strip().replace('\n', ' | ')
    print(f'[{i}] {s[:260]}')
print('SOLU3:', type(SOLU3), getattr(SOLU3, 'shape', None) if not isinstance(SOLU3, dict) else list(SOLU3.keys())[:10])
```