The rate-bin and status rows at index 0 are zero, so those terms don't matter. The remaining error, about 2–3 units, could be noise from the tool's 4-sample average over z. Next I'll measure how much two independent 4-sample local runs differ from each other and from a 64-sample mean. If that spread is similar to 2.4, the local head is effectively exact.

```python
def zh_samples(S_, seed):
    r_ = np.random.default_rng(seed); zs_ = zm[:, None, :] + np.exp(zls)[:, None, :] * r_.standard_normal((len(R), S_, 32))
    return li(zs_, 'z_head', P0)
mA, _ = head_mean2(zh_=zh_samples(4, 1)); mB, _ = head_mean2(zh_=zh_samples(4, 2)); mC, _ = head_mean2(zh_=zh_samples(64, 3))
print('4v4', np.abs(mA-mB).mean().round(3), '4vTool', np.abs(mA-tool).mean().round(3), '64vTool', np.abs(mC-tool).mean().round(3))
print('64vTool by code', {c: np.abs(mC-tool)[cv == c].mean().round(2) for c in [5, 6, 7, 10]}, 'corr', np.corrcoef(mC, tool)[0, 1].round(5))
print('zprior sd mean', np.exp(zls).mean().round(3))
```