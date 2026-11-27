Last result: the decoders designed in run 140 were very poor. A decoder forced to have unit response to the lag-1 dose had a factual output variance about 100 times the innovation floor. So decoding the dose back out of the head input is too noisy to use as the correction. A cleaner place to inject the dose is upstream, at the action token. There the dose is known exactly. The weak point is the policy-centring needed to protect factual fit. Before designing that, I need to recover details my notebook tail does not show: the data layout (are doses hourly START events?), how I estimated the true effects, and what the dosing policy depends on. This run only reads the README, the start of my notebook and the names of my kernel helpers. It is a confirmation step: I expect hourly dose events and an innovation-based truth estimate. If doses are long infusions with a single START, the injection design must change.

```python
nb = open('/work/notes.md').read()
print(open('/data/README.md').read()[:1400])
print('NOTES LEN', len(nb))
print(nb[:3200])
fn = sorted(k for k, v in globals().items() if callable(v) and not k.startswith('_'))
print('FUNCS:', ' '.join(fn)[:900])
```