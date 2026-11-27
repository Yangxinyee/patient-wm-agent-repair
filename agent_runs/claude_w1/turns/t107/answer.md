With perturbed doses, the head input encodes the last reading exactly (R² 1.0) but past doses only partly: lag-1 R² is 0.70, with residual sd 0.78 u/h for insulin and 0.21 for NE. That residual is larger than the natural dose noise in the records (about 0.37–0.65 u/h). So a head-only readout of the dose innovation would add more noise to factual predictions than the signal it captures.

I now think the dose information has to be written into the recurrent state h cleanly, as a dedicated memory of dose innovations with a controlled decay, which the value head then reads. To judge feasibility I need two things: the core parameters that set per-channel decay and input writes, and whether h has near-dead channels (almost constant across states) that could be repurposed with little factual cost. I'll list those tensors and the smallest per-channel standard deviations of h in the factual tuning states `X`.

```python
for n in NAMES:
    if any(s in n for s in ['flow','a_in','delta','b_in','c_out','core.age','out_norm','head.ln','ln_out','z_head','gap']):
        print(n, SHAPES[n])
H = X[:, 192:960].reshape(-1, 4, 192)
sd = H.std(0)        # (4,192)
mu = H.mean(0)
for s in range(4):
    o = np.argsort(sd[s])
    print('slot', s, 'smallest sd ch:', [(int(c), round(float(sd[s,c]),3), round(float(mu[s,c]),2)) for c in o[:6]], 'median sd %.2f' % np.median(sd[s]))
```