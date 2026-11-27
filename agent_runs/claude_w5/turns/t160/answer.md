The dose signal survives into the prelude output. Doubling the insulin dose changes `u` substantially (cos 0.97), the attention output by ~122 and E by 7.5%. But the new state y changes by only 0.5% and `out` by 0.3%. So the loss happens between E and y: the E→h_new write is small next to a very large carried state.

There is a second limit. The head anchors each glucose prediction directly on the last glucose reading (transition table, last-value features), so any state-based shift competes with that anchor. To size an edit I need the head's response to state changes, which means a working local head. The kernel has head helpers from earlier runs (`head_P`, `eval_full`, `head_mean`) of unknown status. I'll run `head_P` with its defaults and inspect what it returns, to see whether it reproduces the fwd_3 predicted means. If it does, I can design and check edits locally. If not, I'll rebuild the head against fwd_3.

```python
import io, contextlib
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        res_hp = head_P(0, 0)
    print("head_P returned:", type(res_hp))
    if isinstance(res_hp, (tuple, list)):
        for i_, v_ in enumerate(res_hp[:6]):
            print(" item", i_, type(v_).__name__, getattr(v_, 'shape', None) if hasattr(v_,'shape') else (str(v_)[:200]))
    elif isinstance(res_hp, dict):
        print(" keys:", list(res_hp.keys())[:20])
    else:
        print(str(res_hp)[:300])
except Exception as ex:
    print("head_P error:", repr(ex)[:300])
printed = buf.getvalue()
print("captured stdout (first 600 chars):", printed[:600])
```