The kernel still has what I need. The value head is W1/b1 (192×192), W2/b2 (17×192) and TR (embed_transition, 13 codes × 17 last bins). SAMP holds the 39,178 perturbed-dose samples and DEC the ridge decoders. There is also a `mvdr(Sig, cons, resp)` helper from last session, and F2df has the tool's factual residuals on setA.

To reuse these correctly I need the exact contracts: how gen_samples perturbs doses and what it returns, what run_rows_wc gives, and what mvdr solves. I'll print their source in truncated form, with nothing hypothesis-testing yet.

```python
import inspect
for fn in ['mvdr','run_rows_wc','head_mean']:
    src = inspect.getsource(globals()[fn])
    print('---',fn,len(src)); print(src[:900])
src = inspect.getsource(gen_samples); print('--- gen_samples', len(src)); print(src[:1800])
```