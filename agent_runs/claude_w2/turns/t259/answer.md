Run 250 explains the run-244 numbers. The 7 cf values are per arm (x2, x1/2, x0, x3, x1.5, x2 from 6, x2 from 16). A ridge decoder of recent insulin dose, read from the head features, moves by only 3–15% of the true dose change when doses are raised (x2, x3, later starts). It moves by 63–96% when doses are lowered.

Conclusion: the original state saturates for doses above the usual range. A head-only readout therefore cannot reproduce "x2". The edit needs a dose channel inside the core so the state carries the actual dose linearly, and that is what `make_core_chan`/`W2` were for. The globals show I analysed the 7-token prelude attention (4 slots + e + u + z, 4 heads × 48 dims) and derived `wI48`/`vI`.

To use that work I need the exact core step and the channel construction. This run prints core_test.py and any stored code strings (`code_txt`, `sig`, `head`), truncated.

```python
print(open('/work/core_test.py').read()[:2400])
for nm_s in ['code_txt','sig','head']:
    v=globals().get(nm_s)
    if isinstance(v,str): print(f"===== {nm_s} ({len(v)})\n", v[:1200])
```