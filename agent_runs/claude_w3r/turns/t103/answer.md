The last result was a re-read of ARCHITECTURE.md and the top of my notes. Nothing new was measured.

- **What it settled:** the value head that predicts the next glucose_finger reading only sees the core state and the same-code register. The insulin dose reaches that state only through a long path: encode_action, then u_k, then the prelude, the recurrent blocks and the coda.
- **Why that blocks a head-only edit:** my earlier probes found the state barely carries the dose's deviation from the policy. So I need a dose channel through the core.
- **Next step:** list the core and encode_action tensors and recall what my local re-implementation (impl.py, impl2.py) already provides. I can then judge whether that code can test locally how a signal injected into the dose path spreads to the head-time state.
- **What would confirm the plan:** impl.py implements a full core step, with prelude, blocks, coda and out_norm, using recognisable tensors. Then I can design and check an injection locally before spending forward passes.
- **What would refute it:** impl.py stops at z_post or the encoders. Then I would have to measure the channel empirically with "states" passes instead.

```python
src=open('/work/impl.py').read(); src2=open('/work/impl2.py').read()
print(len(src), len(src2))
print('impl.py defs:', [l.strip() for l in src.splitlines() if l.startswith('def ') or l.startswith('class ')])
print('impl2.py defs:', [l.strip() for l in src2.splitlines() if l.startswith('def ') or l.startswith('class ')])
print('kernel callables:', [k for k,v in list(globals().items()) if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__'][:120])
for n in NAMES:
    if n.startswith('core') or n.startswith('encode_action') or 'pending' in n or 'filter' in n:
        print(n, tuple(SHAPES[n]))
```