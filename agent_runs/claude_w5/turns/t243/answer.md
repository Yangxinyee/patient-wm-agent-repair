The rollout spec is now clear. Readings at hour 13 and later are drawn from the model and fed back; drug atoms from hour 12 have rate and dose scaled; the readout is the head's predicted mean.

I now suspect my earlier local check was misleading. It probably applied the edit one step at a time from the original model's states, so it missed how the injected belief builds up over the 12 teacher-forced hours. The tool runs the edited model through the whole record, so by 12:05 the state already carries 12 hours of injections. A biased decoded residual, or feedback through c_hat, could push glucose up and MAP down.

To test this I need a fully sequential edited run. First I'll disassemble `run_tf` and `make_bundles` to recover how bundles, times, memory and the step are wired, so I don't guess the conventions.

```python
import dis, io, contextlib
for fn in [run_tf, make_bundles]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dis.dis(fn)
    s = buf.getvalue()
    # compact: keep only opname + argrepr
    lines = []
    for ln in s.splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            lines.append(' '.join(p for p in parts if not p.isdigit() and p not in ('>>',)))
    print(fn.__name__, fn.__code__.co_varnames)
    print('\n'.join(lines)[:3500])
    print('=========')
sid0 = SIDS[0]; b0 = bundles_B[int(sid0)] if int(sid0) in bundles_B else bundles_B[list(bundles_B)[0]]
print(type(b0), len(b0)); print(b0[0]); print(b0[1])
```