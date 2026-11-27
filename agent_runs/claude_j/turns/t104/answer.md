What I now know about the head components:
- `gap` is an MLP from 1 to 192 to 192 inputs.
- `z_prior` maps 192 inputs to 64 outputs, a mean and a log-sd for z.
- The activation is GELU.
- The fwd_2 factual file gives, per reading, the stay, hour, event, observed value, predicted mean and NLL.

That is enough to rebuild the value head on top of the tool's states and check it against those predicted means. Exact agreement is not expected, because the evaluation draws z from the prior four times.

First I need the bundle structure that the earlier session stored in `BUN`. I need each bundle's time, its events, and how bundles line up with the rows of the states file, so that every reading maps to the state of the previous bundle and to its dt. I will print one short excerpt and a few consistency counts.

```python
b0_ = BUN[2020000000]
print(type(b0_[0]).__name__, str(b0_[0])[:300])
print(str(b0_[1])[:300])
print(str(b0_[5])[:300])
print('rows for stay0 in states:', int((sid_ == 2020000000).sum()), 'bundles:', len(b0_))
print('ctx_of consts:', [c for c in ctx_of.__code__.co_consts if not callable(c)][:20])
print('S keys', list(S.keys()) if hasattr(S, 'keys') else type(S))
```