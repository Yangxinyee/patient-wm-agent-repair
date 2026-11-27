The old helper relies on a global mask sized for the full 5,306-row table, so it cannot be called on single rows.

Next I will call it on the full table with an all-zero `zsamp` array, trying the shapes (1, 5306, 32) and then (5306, 32), so that it presumably uses the prior-mean z. I then compare its bin probabilities with my deterministic head for one reading.
- **If the probabilities differ only by a constant shift on the logit scale:** a missing additive term is the cause.
- **If they differ in a bin-dependent way:** an indexing detail is wrong.

```python
p_me = probs_det(r1, ('gelu','gelu'), 1, 'A').ravel()
for shp_ in [(1, 5306, 32), (5306, 32), (5306, 1, 32)]:
    try:
        out_t = head_value2(F_, o_, m_, gv_, zp2_, zsamp=np.zeros(shp_))
        p_old = np.asarray(out_t[1])[i1].ravel()
        print(shp_, 'old pm', np.round(out_t[0][i1],3), 'tool', r1[3])
        print('old', np.round(p_old,3)); print('me ', np.round(p_me,3))
        print('logit diff', np.round(np.log(p_old+1e-12) - np.log(p_me+1e-12), 2))
        break
    except Exception as ex:
        print(shp_, 'ERR', type(ex).__name__, str(ex)[:200])
```