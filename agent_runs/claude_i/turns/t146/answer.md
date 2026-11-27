p(no value) is exactly 0, so the readout formula is not the cause. My head is biased upward by +2.4 mg/dL on average, which points to a systematic detail. It could be the bin indexing of the last value, the code index into the transition table, the status row or the feature layout. The old helper gets these right well enough to be within the tool's noise.

Rather than keep guessing, I will probe the old helper as a black box. I call it on single readings with a fixed z input (`zsamp` of zeros), so it should use the prior mean. Then I compare its bin probabilities with my deterministic version given exactly the same inputs (ReLU gap, prior-mean z).
- **If they agree on most bins but not all:** the bin indexing is off.
- **If they are shifted in logit space:** an additive term such as status or the feature layout is wrong.
- **If the call fails:** the error message will reveal the expected z format.

```python
i1 = idx_chk[1]; r1 = rows_chk[1]
try:
    out_t = head_value2(F_.loc[[i1]], o_[[i1]], m_[[i1]], gv_[[i1]], zp2_[[i1]], zsamp=np.zeros((1,1,32)))
    p_old = np.asarray(out_t[1]).ravel()
    p_me = probs_det(r1, ('gelu','gelu'), 1, 'A').ravel()
    print('old pm', np.round(out_t[0],3), 'tool', r1[3])
    print('old', np.round(p_old,3)); print('me ', np.round(p_me,3))
    print('logit diff', np.round(np.log(p_old+1e-12) - np.log(p_me+1e-12), 2))
except Exception as ex:
    print('ERR', type(ex).__name__, str(ex)[:400])
```