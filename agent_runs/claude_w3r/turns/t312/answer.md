Results from the last run:
- The new gf-only gate separates cleanly on held-out rows (margin 1.77) and is orthogonal to both drug-signal directions.
- The refit rate reads have held-out noise sd 0.28 for insulin and 0.04 for norepinephrine, with baselines 55.56 and 11.11. That noise level is acceptable.
- The activation check failed: my numpy version of the value head is off by up to 110 logits from `HD.logits(HD.hidden(...))`. So the head probably applies something before the first layer, such as a LayerNorm. The pre-activation and variance numbers above therefore cannot be trusted yet.

Next, I will disassemble `HD.hidden`, `HD.logits`, `HD.MLP`, `HD.L` and `HD.LN` and look at `HD.W`, to find the exact computation. If there is a norm before `value_bin.0`, I will refit the reads in the space after that norm.

```python
import dis
for m in ['hidden', 'logits', 'MLP', 'L', 'LN']:
    f = getattr(HD, m); f = getattr(f, '__func__', f)
    print(m, ' '.join(f"{i.opname[:12]}:{i.argrepr}" for i in dis.get_instructions(f) if i.opname not in ('RESUME', 'CACHE', 'PRECALL'))[:700])
print(type(HD.W).__name__, (list(HD.W.keys())[:12] if isinstance(HD.W, dict) else ''), len(HD.W) if hasattr(HD.W, '__len__') else '')
print([k for k in (HD.W.keys() if isinstance(HD.W, dict) else []) if 'value' in k or 'with' in k or 'norm' in k][:20])
```