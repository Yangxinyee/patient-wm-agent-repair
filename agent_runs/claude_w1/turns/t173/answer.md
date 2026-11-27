This is the decisive offline comparison. For insulin, the head-level correction (decoded lag-1 dose minus the last-reading-bin mean, c = −2.5) gives a predicted net factual gain (−0.0004 nats overall). The bin-mean centring even beats centring on the true policy, because it also removes context bias in the decoder. So the head-level route works for insulin.

For NE it costs +0.0018 nats at c = 3 for lag 1 alone. The decoder's factual error sd is 0.21 mcg/kg/min against an innovation sd of 0.08. That is too noisy once the larger multi-lag effect and the insulin cost are added.

To clean up the NE signal I will try a relay. An action-token injection, exact in the dose, pushes a dose signal into the head input, ideally along directions where the head input varies little naturally, and a head unit then reads it. This needs each head-2 basis direction's effect on the full head input, not just on the predicted mean. So I rerun the sweep storing the head-input changes (as float32, about 110 MB).

```python
def run_rows_wc(stays):
    rows = [r for s in stays for r in gen_samples(G3[s], pert=False)]
    res = head_mean(rows); mu = [np.asarray(x) for x in res if np.ndim(x)==1 and len(x)==len(rows)][0]
    return np.stack([r[8] for r in rows]), mu
t0 = time.time(); WC0, mu0b = run_rows_wc(H100); print('base check', np.abs(mu0b-mu0).max())
DWC = np.zeros((48,) + WC0.shape, dtype=np.float32)
for i in range(48):
    W = KV0.copy(); W[288+i] += eps*(w_ins + w_ne).astype(KV0.dtype); Wo['core.prelude.0.attn.kv.weight'] = W
    wci, _ = run_rows_wc(H100); DWC[i] = ((wci - WC0)/eps).astype(np.float32)
Wo['core.prelude.0.attn.kv.weight'] = KV0.copy()
np.save('/work/sweep_dwc.npy', DWC); print('secs %.0f' % (time.time()-t0), 'mean |dwc| per basis %.4f' % np.abs(DWC).mean())
```