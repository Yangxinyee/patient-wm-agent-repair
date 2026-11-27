The state barely responds to z. A unit change in z moves `out` by about 0.05, against a natural spread of 5.3, so the z route into the state is nearly closed. There are many dead units on the head side (127 in z_prior, 10 in the value head), but they can only read what ctx contains. The dose deviation is barely there.

So the dose has to be written into the recurrent state more strongly, through the prelude, the only place u reaches the slots. Whether that can persist depends on two things:
- whether `c_out` is close to identity, giving natural decaying memory channels;
- how much the slots attend to the u token.

I'll check `c_out`'s distance from identity, the spread of its eigenvalues, the per-dimension decay timescales, and the prelude tensor names. A near-identity `c_out` with some slowly decaying dimensions would offer a memory channel. A strongly mixing `c_out` would not.

```python
Co = T64('core.c_out.weight').numpy()
print('c_out: ||C-I||/||I||', (np.linalg.norm(Co-np.eye(192))/np.sqrt(192)).round(3), 'diag mean', np.diag(Co).mean().round(3), 'offdiag rms', np.sqrt(((Co-np.diag(np.diag(Co)))**2).mean()).round(4))
ev_c = np.linalg.eigvals(Co); print('|eig| quantiles', np.quantile(np.abs(ev_c),[0,.1,.5,.9,1]).round(3))
fa_n = T64('core.flow.a').numpy(); dl = T64('core.delta_in').numpy(); ai = T64('core.a_in').numpy()
abar = np.exp(-np.log1p(np.exp(dl))*np.exp(ai))
tau_flow_h = 1/np.exp(fa_n)/3600
print('flow timescale (h) quantiles', np.quantile(tau_flow_h,[0,.1,.5,.9,1]).round(2), '| a_bar quantiles', np.quantile(abar,[0,.1,.5,.9,1]).round(3))
print([n for n in NAMES if n.startswith('core.prelude') or n.startswith('core.blocks.0.0')][:40])
```