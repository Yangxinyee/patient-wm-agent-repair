The cost does not fall with the gain, so most of it is not the intended dose effect. At s = 0.3 the value NLL rises +0.173, against +0.254 at s = 1:

| Reading | Cost at s = 1 | Cost at s = 0.3 |
|---|---|---|
| Finger glucose | +0.166 | +0.038 |
| Lab glucose | +0.393 | +0.133 |
| Arterial MAP | +0.374 | +0.301 |
| Cuff MAP | +0.175 | +0.225 (higher) |

So most of the damage, especially to MAP, is unrelated to the injection gains.

**Hypothesis.** The prelude still writes the channel into the prelude output E at a fixed scale: about 6 × rate / σ along q. That is hundreds of natural standard deviations, and it does not shrink with the b_in gain. If b_in is not the only tensor reading E (for example a step-size layer such as delta_in), every such reader sees a large out-of-distribution input at every drug bundle. That would damage the state regardless of gain.

**What this run checks:**
1. Whether the fwd_13 prediction shifts scaled down with the gain. Costs without shifts would point to shape or scale damage.
2. Which tensors consume E, from the architecture notes.
3. How strongly each core matrix with 192 inputs responds to the two channel directions, compared with its natural variation.

**How to read it.** A strong response in some E-reader other than b_in would confirm the hypothesis. The fix is then to project the channel directions out of those readers, which costs nothing in normal operation because natural activity there is tiny. If no other reader responds, the cost lies elsewhere.

```python
fe=pd.read_csv('/work/fwd_13/factual_edited.csv'); fo=pd.read_csv('/work/fwd_13/factual_original.csv')
d13=pd.DataFrame({'event':fe.event,'dp':fe.predicted_mean-fo.predicted_mean,'dn':fe.nll-fo.nll})
print(d13.groupby('event').agg(dp_mean=('dp','mean'),dp_rms=('dp',lambda x: np.sqrt((x**2).mean())),dn=('dn','mean'),dn_med=('dn','median')).round(3))
txt=open('/model/ARCHITECTURE.md').read().splitlines()
hits=[l.strip()[:150] for l in txt if any(s in l for s in ['prelude_norm','b_in','delta','c_out','gate','E_t',' E '])]
print('ARCH hits:',len(hits)); print('\n'.join(hits[:22]))
Enat=f64(Z['E']).reshape(-1,192); qEI=pnw*V4['qI']; qEN=pnw*V4['qN']
for n in NAMES:
    if not n.startswith('core.') or n.startswith('core.prelude'): continue
    sh=tuple(SHAPES[n])
    if len(sh)!=2 or sh[1]!=192: continue
    W=f64(load_original(n).numpy()); Y=Enat@W.T
    sdn=np.sqrt(((Y-Y.mean(0))**2).mean())
    print(n,sh,'resp qI %.3f qN %.3f | nat sd %.3f'%(np.linalg.norm(W@qEI)/np.sqrt(sh[0]),np.linalg.norm(W@qEN)/np.sqrt(sh[0]),sdn))
```