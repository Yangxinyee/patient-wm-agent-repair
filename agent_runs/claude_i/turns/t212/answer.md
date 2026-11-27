The reproduction matches the tool's states exactly at the bundles I checked (z and `out` errors both 0). The rebuilt bundle table has 1,362 insulin starts, 1,612 norepinephrine starts and 1,749 glucose_finger readings.

So I can compute the effect of a z change at a drug bundle on the next reading's predicted mean locally, by finite differences. I will do this for:
- 200 insulin bundles, with glucose_finger next;
- 200 norepinephrine bundles, with MAP next.

Then I will find the z direction that moves the target reading while cancelling the average effect on the other readings. I will also print how consistent that effect is across bundles, and save the raw derivatives.

The z route is confirmed if, along that direction, the target derivative's mean is large compared with its spread and the side effects stay near zero. If the spread is as large as the mean, the injected effect would vary wildly between patients, and I would look for a different place to inject.

```python
import time
def head_mean(o1, h1, k1, code):
    cx = ctx_of_state(o1.double()[None], h1.double()[None], float(dts[k1])).double()
    lv = LVr[code][k1]; la = (INFO[k1]['t'] - LTr[code][k1])*60.0 if lv==lv else 0.0
    m,_ = value_mean(cx, [code], np.array([lv]), np.array([la]))
    return float(m[0])
def jac_at(k, codes, eps=0.2):
    hf = HHt[k-1].to(fa_.dtype)*torch.exp(-dts[k].to(fa_.dtype)*torch.exp(fa_))
    cx = ctx_from(OUTt[k-1].to(fa_.dtype), hf, dts[k]); z0 = zpost_mean(cx, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype))
    j0 = int(np.where(sid3==sid3[k])[0][0]); im = np.arange(max(j0,k-127), k+1); ag = torch.as_tensor((tmin[k]-tmin[im])*60.0, dtype=fa_.dtype)
    Jm = np.zeros((32, len(codes)))
    h1,o1 = core_step(hf, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype), z0, Ee[im].to(fa_.dtype), ag, V)
    b = [head_mean(o1,h1,k+1,c) for c in codes]
    for j in range(32):
        zz = z0.clone(); zz[j] += eps
        h2,o2 = core_step(hf, Ee[k].to(fa_.dtype), Uu[k].to(fa_.dtype), zz, Ee[im].to(fa_.dtype), ag, V)
        Jm[j] = [(head_mean(o2,h2,k+1,c)-bb)/eps for c,bb in zip(codes,b)]
    return Jm
codes4 = ['glucose_finger','map_arterial','map_cuff']
okI = [k for k in range(1,NB-1) if 'insulin' in INFO[k] and sid3[k]==sid3[k-1]==sid3[k+1] and 'glucose_finger' in INFO[k+1]]
okN = [k for k in range(1,NB-1) if 'norepinephrine' in INFO[k] and sid3[k]==sid3[k-1]==sid3[k+1] and 'map_arterial' in INFO[k+1]]
rs = np.random.RandomState(0); sI = rs.choice(okI, 150, replace=False); sN = rs.choice(okN, 150, replace=False)
t0=time.time(); JI = np.stack([jac_at(int(k), codes4) for k in sI]); JN = np.stack([jac_at(int(k), codes4) for k in sN])
np.save('/work/JI.npy', JI); np.save('/work/JN.npy', JN); np.save('/work/sIN.npy', np.stack([sI,sN]))
print('time', round(time.time()-t0), 'n', len(okI), len(okN), '| z sd mean', round(float(ZZt.std(0).mean()),3))
def best_dir(Jall, tgt, side):
    mt = Jall[:,:,tgt].mean(0); S = np.stack([Jall[:,:,s].mean(0) for s in side],1)
    v = mt - S @ np.linalg.lstsq(S, mt, rcond=None)[0]; v /= np.linalg.norm(v)
    eff = Jall @ v
    return v, [(round(eff[:,c].mean(),3), round(eff[:,c].std(),3)) for c in range(Jall.shape[2])]
vI, eI = best_dir(JI, 0, [1,2]); vN, eN = best_dir(JN, 1, [0,2])
print('INS dir effects (mean,sd) on gf,ma,mc:', eI, '| |mean dGF|', round(np.linalg.norm(JI[:,:,0].mean(0)),3))
print('NE  dir effects (mean,sd) on gf,ma,mc:', eN, '| |mean dMA|', round(np.linalg.norm(JN[:,:,1].mean(0)),3))
```