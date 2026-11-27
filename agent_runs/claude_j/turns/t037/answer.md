The model's dose effect is too weak from the first hour, not just in how it carries forward. One hour after the first edited dose it gives 26% of the true insulin effect and 11% of the true norepinephrine effect. Later lags stay at about 16–25% of the truth for insulin. So the edit must make the dose itself move the next glucose/MAP prediction much more.

How to do that without harming factual fit depends on the model's internals. I'll list all 181 tensors with shapes. That tells me how the action encoder, prelude, recurrent update (b_in, delta_in, a_in, flow.a), coda and value head are built. It also tells me whether I can re-implement the forward pass on CPU and check it against the tool's outputs.

```python
for n_ in NAMES:
    print(n_, tuple(SHAPES[n_]))
print(json.dumps(VOCAB.get('config', {}))[:1500])
```