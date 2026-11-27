`step_at` returns two values, not three. Before rerunning the loop I'll call it once and print the shapes of what it returns. If E isn't among them, I'll use `step_parts`, whose locals include E, x, hn, y and which returns 5 values. Success means I can get E for one drug bundle.

```python
s0 = int(sorted(recs.stay_id.unique())[0]); B_, rows_, EU_ = stay_data(s0)
ks = [k for k in range(1,len(B_)) if any(a['code'] in (ci,cn) and a['kind']==0 for a in B_[k][1])]
r1 = step_at(B_, rows_, EU_, ks[0])
print('step_at ->', [ (type(x).__name__, tuple(getattr(x,'shape',()))) for x in r1])
print('step_parts globals rows0/B0/EU0 refer to a fixed stay; signature', inspect.signature(step_parts))
print('core_step_v ->', core_step_v.__code__.co_varnames[:12])
```