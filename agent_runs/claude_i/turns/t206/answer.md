The architecture document clears up three things:
- The state update uses the posterior mean of z. That posterior is computed from [ctx, e_k, u_k], so it sees the drug atoms (u_k) and the context in the same step.
- Drug START atoms reach the model only through u_k, and u_k is not stored in the memory M. The memory M holds only measurement and END encodings, which carry no rate.
- The value head predicts a softmax over bins. Its predicted mean is the sum of bin centres weighted by their probabilities.

So the most promising place to inject the dose effect is the 50 dead hidden units of `z_post`. They can compute the dose deviation (rate minus what the policy would give in this state) from u_k and ctx, and write it into z. This changes the state only when a dose departs from the policy, which should keep factual predictions intact.

Before designing it, I need the exact shapes of the z_post, z_in, z_prior, z_head and head tensors, and a reminder of my helper files (`sim.py`, `head2.py`). I will judge the plan workable if z_post is a single-hidden-layer MLP whose dead units I can address. It is ruled out if z_post has another structure, for example more than one hidden layer with no dead units.

```python
tj = json.load(open('/model/tensors.json'))
for n in NAMES:
    if n.startswith('core.blocks') or n.startswith('core.prelude.'): continue
    print(n, SHAPES[n])
print('---- sim.py ----'); print(open('/work/sim.py').read())
print('---- head2.py ----'); print(open('/work/head2.py').read()[:2800])
```