This is a serious mismatch. On setA rows the decoders' outputs are offset by about -215 (insulin) and -36 (NE), with within-cell spread 10-30 times the train-replica estimate. The setA head inputs (wc_, built from tool states) and the train-replica head inputs (Xf, built by q_of) therefore differ systematically, and the high-norm minimum-noise decoders amplify the difference.

Decoders must be built and checked on one consistent construction, ideally the one closest to the real model. I will check q_of's signature and apply it to the setA tool states (Xp) to see which construction wc_ matches. Close agreement means the offset comes from elsewhere (for example z handling). A large difference means one construction is wrong and I must find out which.

```python
print(inspect.signature(q_of), q_of.__code__.co_names[:30])
print(inspect.signature(run_head), run_head.__code__.co_names[:30])
```