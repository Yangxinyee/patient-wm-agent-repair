The leak does not come from the "no drug" token: the edited channel readout there is about 0.002, effectively zero. It must enter where the edited input map `b_in` reads the channel direction of the processed input.

I chose those channel directions to be quiet at drug bundles only. Reading bundles have different inputs, so the channel coordinate may be far from zero there. `b_in` then turns that natural value into a large injection at every reading bundle, whether or not any drug was given. To get the required size, my scaling implies a channel coordinate of about −0.2 at reading bundles, against a natural spread of 0.0008 at drug bundles.

The kernel still has `prelude_b(T, Wkv, Wo)`, a CPU copy of the input-processing block, plus stored reading-bundle tokens `Tm`. I also learned that `build_v4` has extra arguments `cent` and `save`, and found an unfinished v5 design whose targets use the level directions from run 265.

This run will:
- check `prelude_b` against the stored drug-bundle results;
- compute the channel coordinate at reading bundles with the original weights;
- convert it into the implied per-bundle push in mg/dL.

How to read it:
- A push of roughly +1 to +3 mg/dL per bundle, with the right sign, confirms the leak mechanism. The fix is then channel directions quiet at both bundle types.
- A tiny value refutes it and sends me to a forward test with parts of the edit removed.

```python
import numpy as np
print('build_v4 defaults', build_v4.__defaults__, ' c', c, ' rbar', rbar, ' kv3', getattr(kv3,'shape',None), ' gp', getattr(gp,'shape',None))
rd = np.load('/work/rep_drug.npz')
o = prelude_b(Td[:40], Wkv, Wo)
outs = o if isinstance(o, (tuple, list)) else (o,)
for i, oo in enumerate(outs):
    oo = np.asarray(oo); print('prelude_b out', i, oo.shape)
    for key in ['x2', 'E', 'x1']:
        ref = rd[key][:40]
        if oo.shape[-2:] == ref.shape[-2:] or (oo.ndim==3 and oo.shape[1]>=4):
            cand = oo[:, :4] if oo.shape[1] > 4 else oo
            if cand.shape == ref.shape: print('   vs', key, 'max abs err', np.abs(cand-ref).max().round(4), ' ref norm', np.linalg.norm(ref,axis=-1).mean().round(2))
def chan_coord(X):  # X: (n,4,192) prelude output x'' at slot tokens -> s = q . LN(x'')
    mu = X.mean(-1, keepdims=True); sd = X.std(-1, keepdims=True)
    return ((X-mu)/np.sqrt(sd**2+1e-5)) @ qs, sd[...,0]
xd = rd['x2']; sD, sigD = chan_coord(xd)
print('DRUG bundles  s mean', sD.mean((0,1)).round(5), ' sd', sD.std((0,1)).round(5), ' sigma median', np.median(sigD).round(1))
om = prelude_b(Tm, Wkv, Wo); om = np.asarray(om[0] if isinstance(om,(tuple,list)) else om)
xm = om[:, :4]
sM, sigM = chan_coord(xm)
print('MEAS bundles  s mean', sM.mean((0,1)).round(5), ' sd', sM.std((0,1)).round(5), ' pct5/50/95 chI', np.percentile(sM[...,0],[5,50,95]).round(4), ' chN', np.percentile(sM[...,1],[5,50,95]).round(4))
sr = float(np.load('/work/v4_parts.npz')['sig_ref'])
print('sig_ref', sr, ' leak in f-units per bundle (s*sig_ref/c): mean', (sM.mean((0,1))*sr/c).round(3), ' -> insulin news mg/dL approx', (-0.87*sM[...,0].mean()*sr/c).round(3), ' NE news mmHg approx', (4.3*sM[...,1].mean()*sr/c).round(3))
```