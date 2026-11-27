The old encoder outputs e = 0 when a bundle has no measurement atoms and u = 0 when it has no actions. Activations defaulted to ReLU. I'm unsure about the activation, the unit indices and the time units. I'll pin down the time units first: if exp(-core.flow.a) runs from about 600 to 6×10^7, the documented 10 minutes to 2 years, then dt is in seconds. I'll also look at the gap MLP's input scale, softplus(delta_in) and a_bar. These fix the time-flow part of ctx before I re-check the z posterior.

```python
fa_ = load('core.flow.a').numpy().astype(np.float64)
ts = np.exp(-fa_)
print('timescale range (units of 1/exp(a)):', ts.min(), np.median(ts), ts.max())
print('in hours if seconds:', ts.min()/3600, ts.max()/3600)
d_in = load('core.delta_in').numpy().astype(np.float64); a_in_ = load('core.a_in').numpy().astype(np.float64)
sp = np.log1p(np.exp(d_in)); abar = np.exp(-sp*np.exp(a_in_))
print('softplus(delta_in) range', sp.min(), sp.max(), 'abar range', abar.min(), np.median(abar), abar.max())
```