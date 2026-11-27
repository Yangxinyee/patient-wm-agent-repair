The architecture answers two key questions.

- **The head has no dose information of its own.** The "last" register inputs refer to the most recent earlier value of the same code. So a glucose head sees only the last glucose reading. All insulin history has to come through `out`, the time-flowed h and z.
- **Why my re-implementation is off.** At evaluation the head's z is drawn 4 times from the prior and the probabilities are averaged. My re-implementation most likely used the posterior or prior mean. That would explain the NLL gap from run 71 (about 0.02–0.07 too high).

What this means for the edit:
- **Head-only correction from existing state.** A linear decoder of dose deviation reaches only R² ≈ 0.46. It would likely under-respond to doubled doses in a rollout, by roughly that factor.
- **A dedicated dose memory.** The true insulin response builds up and fades over about 10 h, and the model would need to carry that. The only memory that persists across bundles is h. It is decayed per dimension by `core.flow.a`, mixed by `core.c_out` and passed through `core.coda` at every step.

Next I look at the structure of `core.*`: the shapes, what `coda` is (a norm? an MLP?), the decay timescales, and the per-dimension spread of h across bundles in the fwd_3 states. Low-variance (near-dead) dimensions with suitable timescales would make a dedicated memory channel feasible at low factual cost. If every dimension is busy and `coda` normalises strongly, I would fall back to a head-side decoder with an attenuation correction.

```python
import numpy as np, torch
core = [n for n in NAMES if n.startswith('core.')]
for n_ in core: print(n_, SHAPES[n_])
fa = load('core.flow.a').flatten().numpy()
tau = 1/np.exp(fa)  # seconds
print('flow timescale (h) quantiles:', np.round(np.quantile(tau/3600, [0,.1,.25,.5,.75,.9,1]),3))
dl = load('core.delta_in').flatten().numpy(); ai = load('core.a_in').flatten().numpy()
sp = np.log1p(np.exp(dl))
abar = np.exp(-sp*np.exp(ai))
print('softplus(delta_in) q:', np.round(np.quantile(sp,[0,.5,1]),3), 'a_bar q:', np.round(np.quantile(abar,[0,.1,.5,.9,1]),3), abar.shape)
Hs = HH.numpy() if hasattr(HH,'numpy') else HH
sd = Hs.std(0)  # (4,192)
print('per-slot std quantiles:', [np.round(np.quantile(sd[s_],[0,.05,.5,1]),3) for s_ in range(4)])
lo_dims = np.argsort(sd.mean(0))[:12]
print('lowest-std dims:', lo_dims, np.round(sd.mean(0)[lo_dims],4), 'timescale h:', np.round(tau[lo_dims]/3600,2) if tau.shape[0]==192 else tau.shape)
```