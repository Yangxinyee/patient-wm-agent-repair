Run 210 confirms that "expF" is the factual log-loss change on glucose readings (positive means worse). At penalty 1000 it is +0.021 nats per glucose reading, about +0.007 per reading overall, which is still over the limit. That penalty recovers only about 25% of the true effect. So a readout from the head features alone carries too little dose information. The model's state at prediction time encodes the dose weakly.

The kernel also holds `set_amp` and `amp_eval`, which suggests I once tested amplifying the dose input. Next I will search my notebook for any record of that test.

```python
s=open('/work/notes.md').read()
import re
L=[l for l in s.split('\n') if re.search(r'amp|NW0|expF|frontier|gain|readout',l,re.I)]
print(len(s),len(L)); print('\n'.join(x[:160] for x in L[:14]))
```