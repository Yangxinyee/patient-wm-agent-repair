The value head reads `with_code`, which is `q` plus the code embedding plus a "last value of the same code" register. Its logits also get a transition table indexed by code and last bin. That register explains the model's glucose persistence. The only parameters that touch value predictions without feeding back into the recurrence are the value-head MLP, the transition table and the register embeddings. A dose-dependent, code-specific correction therefore has to go through the MLP's hidden units, for example by adding a rank-1 term to the first layer's weights. Choosing that term needs an accurate offline copy of `with_code`, and run 107's errors suggest my copy was not exact. Next I'll inspect my existing `head`/`headv` helpers to see what they reproduce.

```python
import inspect
for fname in ['head','headv','prep']:
    try:
        s = inspect.getsource(globals()[fname]); print('----',fname,len(s)); print(s[:1400])
    except Exception as ex: print(fname, ex)
```