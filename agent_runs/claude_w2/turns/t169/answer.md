The file listing gives me the APIs. `run_rollout(R, W, arms, ...)` takes a weight dict `W`, so I can test any edited weights locally. Its `store` argument collects the head inputs (base, probabilities, last-value features, z, h, out) at every measurement. That is what I need to fit a decoder for the head correction. Next I print the rest of rollout.py (pooling, core step, readout) and head.py and tf.py, so I can add a correction hook and compute local teacher-forced NLL.

```python
s=open('/work/rollout.py').read()
print(s[7000:])
print('===== head.py'); print(open('/work/head.py').read())
print('===== tf.py'); print(open('/work/tf.py').read())
```