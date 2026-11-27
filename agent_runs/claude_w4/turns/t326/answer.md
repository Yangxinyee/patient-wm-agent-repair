The prediction shifts at s = 0.3 confirm that most of the factual damage does not depend on the gain:

| Reading | Shift rms, s = 1 | Shift rms, s = 0.3 | Gain-independent part |
|---|---|---|---|
| Finger glucose | 15.3 | 9.5 | ≈ 8.7 mg/dL |
| Lab glucose | 21.7 | 16.2 | ≈ 15.5 mg/dL |
| Arterial MAP | 6.6 | 6.4 | ≈ 6.4 mmHg |
| Cuff MAP | 5.7 | 6.7 | ≈ 6.7 mmHg |

Fitting a gain-independent part plus a part that scales with the gain gives the last column. The median NLL change is about 0.001, so the cost comes from a minority of readings hit hard.

**Where the damage enters.** The architecture's state update is `h <- a_bar*h + softplus(delta_in)*b_in(E) + …`, and the step size `delta_in` is a constant parameter. No core matrix reacts unusually to the channel directions. That leaves the channel's own write into the prelude, which is fixed at scale c = 6 (about 6·rate/σ along q) whatever the b_in gain. Through the prelude FFN, or E feeding h directly, that write disturbs the state at every drug bundle.

**Fix.** Cut c to 0.4 and raise the b_in gain by 6/0.4 = 15. The intended b_in response is then exactly the same, while the stray write is 15× smaller. The extra noise from natural activity along q rises to only about 0.16 rate units.

I test v9 at insulin gain 1.0 and NE gain 2.0, both on the v7 scale, with:
- a factual pass;
- a dose_edit pass.

I also print the lines around the update formula to see how E reaches h.

**Expected outcome.**
- The arterial-MAP cost should fall from +0.30 to under about +0.005.
- What remains is the gain-dependent insulin cost, about 0.004–0.008.
- Dose effects: insulin x2 near −2.9 mg/dL and NE x2 near +1.1 mmHg.

If the MAP cost stays near +0.3, this hypothesis is wrong.

```python
txt=open('/model/ARCHITECTURE.md').read().splitlines()
ix=[i for i,l in enumerate(txt) if 'b_in(E)' in l]
for i in ix[:1]:
    for l in txt[max(0,i-3):i+4]: print('|',l.strip()[:160])
def build8(B,tIv,tNv,kI,kN,cc,rbI=float(V4['rbar'])):
    kv=f64(kv0).copy(); kv[240:288]+=A_@M_@B.T
    Wo=f64(Wo0).copy(); Q3=np.stack([V4['qI'],V4['qN'],-rbI*V4['qI']],1)
    Wo[:,48:96]=f64(Wo0[:,48:96])@(np.eye(48)-A_@A_.T)+cc*Q3@Mi_@A_.T
    Wb=Wb0.copy(); sc=c_/cc
    for k,q,t in [(kI,V4['qI'],tIv),(kN,V4['qN'],tNv)]: Wb+=np.outer(k*sc*t-Wb0@(pnw*q),q/pnw)
    return kv,Wo,Wb
kI,kN,cc=1.0,2.0,0.4
kvn,Won,Wbn=build8(B,V5['tI5'],V5['tN5'],kI,kN,cc)
for n,x in [('core.prelude.0.attn.kv.weight',kvn),('core.prelude.0.attn.out.weight',Won),('core.b_in.weight',Wbn)]:
    save_edit(n,torch.tensor(x,dtype=torch.float32))
print('saved v9 kI %.2f kN %.2f c %.2f'%(kI,kN,cc))
with open('/work/notes.md','a') as f:
    f.write('- fwd_13 = factual both sel100 v7(1.455,2.07): NLL +.173 (finger +.038 lab +.133 art +.301 cuff +.225). dp rms 9.5/16.2/6.4/6.7 (s=1: 15.3/21.7/6.6/5.7)\n'
            '  => large GAIN-INDEPENDENT perturbation (finger ~8.7, lab ~15.5, art ~6.4 rms); dn median ~.001 (heavy tail). Gain-dep part: finger ~12.6*s, lab ~15.2*s.\n'
            '  Hypothesis: channel injection magnitude c=6 into prelude x1/E (fixed) disturbs state. Fix: c->0.4, b_in gain x15 (same b_in response).\n'
            '- Run 287: build8(B,tI5,tN5,kI,kN,cc). SAVED v9 = build8(...,1.0,2.0,0.4). Next: factual + dose_edit sel100.\n')
```