The tool and my offline replica agree on the edited model. On the same 40 calibration stays:

| Readout | Tool | Offline |
|---|---|---|
| insulin ×2 | −14.66 | −14.93 |
| insulin ×½ | +6.83 | +7.19 |

So the replica runs about 2–5% high, but it is trustworthy. The undershoot on all 100 stays comes from the other 60, which respond much less: −10.3 for ×2 and +4.82 for ×½. My truth estimate for those 60 is about −13.8 for ×2. So this is heterogeneity between stays, not a mismatch between tool and replica (explanation (a)). The 40-stay calibration set simply responded more strongly than average.

The evaluation is a pooled mean over unseen stays, so I should calibrate on all 100 stays. My kernel-only functions have no source, so I will use the cleanest strength control the readout has. That is to scale the output columns of the two readout units (107 = signal, 109 = gate reference) by a factor f, which preserves their mutual cancellation.

This run does four things:
1. It prints the readout columns and bias deltas, to check this scaling is sound.
2. It splits the edited-versus-truth gap by hour and by glucose level, to see where the undershoot sits.
3. It tests whether the saved cost function reproduces the last run's costs (+0.0006 and +0.0001).
4. It rolls out offline on all 100 sel1 stays at f = 1.2 and 1.35.

Expected outcome:
- **Confirmed if** f ≈ 1.25–1.3 brings the offline ×2 readout to about −14.3 × 1.03 ≈ −14.8, matching the replica's known bias, with a factual cost still below about +0.001.
- **Rethink needed if** the factual cost rises sharply, which would mean the bias holds a readout offset that scaling breaks, or if the gap is concentrated in later hours, which would mean the memory decays too fast.

```python
import torch, numpy as np, pandas as pd, time, json, inspect
T0=time.time()
key2='attr_heads.value_bin.2.weight'; keyb='attr_heads.value_bin.2.bias'
Pcur = {n: load(n) for n in NAMES}
W2 = Pcur[key2]; W2o = load_original(key2); b2 = Pcur[keyb]; b2o = load_original(keyb)
print(tuple(W2.shape), 'changed cols', torch.nonzero((W2-W2o).abs().sum(0)>1e-7).flatten().tolist())
print('c107', np.round(W2[:,107].numpy(),3).tolist()); print('c109', np.round(W2[:,109].numpy(),3).tolist())
print('db2', np.round((b2-b2o).numpy(),3).tolist())
for f_ in [dcost, cost, nll_of]:
    try: print(f_.__name__, inspect.signature(f_))
    except Exception as ex: print('sig', ex)
# per-row truth diagnostics on fwd5
de = pd.read_csv('/work/fwd_5/dose_edit_edited.csv')
r = de[de.reached & de.edit.str.startswith('insulin')].copy()
try:
    t0=time.time(); v = true_shift(int(r.stay_id.iloc[0]), int(r.hour.iloc[0]), r.edit.iloc[0]); print('true_shift ok', v, round(time.time()-t0,3))
    r['truth'] = [true_shift(int(s), int(t), e) for s, t, e in zip(r.stay_id, r.hour, r.edit)]
    for ed in ['insulin_x2','insulin_half']:
        q_ = r[r.edit==ed].copy()
        print(ed, 'pooled edited', round(q_['shift'].mean(),2), 'truth', round(q_.truth.mean(),2))
        gh = q_.groupby('hour')[['shift','truth']].mean().round(1)
        print(' hours', gh.index.tolist(), 'ed', gh['shift'].tolist(), 'tr', gh['truth'].tolist())
        q_['fb'] = pd.qcut(q_.factual, 4, labels=False)
        gf = q_.groupby('fb')[['factual','shift','truth']].mean().round(1)
        print(' factual-quartiles G', gf.factual.tolist(), 'ed', gf['shift'].tolist(), 'tr', gf['truth'].tolist())
except Exception as ex:
    print('truth diag err', repr(ex)[:300])
def scaledP(f):
    P = dict(Pcur); W = W2.clone(); W[:, [107,109]] = f*W2[:, [107,109]]; P[key2] = W; return P
try:
    t0=time.time(); print('dcost(Pcur)', dcost(Pcur), round(time.time()-t0))
except Exception as ex: print('dcost err', repr(ex)[:300])
for f in [1.2, 1.35]:
    if time.time()-T0 > 480: print('skip', f); continue
    t0=time.time(); Pf = scaledP(f); R = eval_shifts(Pf, sel['sel1'], 20, 0); print('f', f, R, 'sec', round(time.time()-t0))
    try: print('  dcost', dcost(Pf))
    except Exception as ex: print('  dcost err', repr(ex)[:200])
print('total sec', round(time.time()-T0))
```