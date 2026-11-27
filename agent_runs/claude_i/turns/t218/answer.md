The regression supports the premise. The original model's teacher-forced errors are explained by past dose deviations that it ignores:

| Effect | Lag 1 | Lag 2 | Lag 3 | Lag 4 | SE | Later lags |
|---|---|---|---|---|---|---|
| Insulin → glucose_finger, mg/dL per unit/h | −1.2 | −1.9 | −1.2 | — | ≈0.55 | noise |
| Norepinephrine → MAP, mmHg per mcg/kg/min | +1.3 | +1.3 | +2.7 | +2.3 | ≈0.85 | — |

So adding the effect of dose deviations should leave factual fit unchanged or slightly better. The estimates are noisy (100 stays), so the final size will come from the tool's dose-edit rollouts, checked against the targets I derived earlier from the records.

Next I build the training set for the readout:
- Re-run the 100 training stays through my exact state-update copy with insulin and norepinephrine rates rescaled by random factors (0 to 3) from random start hours.
- Keep the head's normalised input at each reading.
- Use as targets the lagged dose deviations under the edited doses.

First the run checks that my encoder reproduces the tool's drug encodings exactly. Then a ridge regression shows how well the head input encodes each lag's deviation.

The plan holds if the insulin lags 1–2 come out with held-out R² of roughly 0.7 or more. A low R² would mean the state does not carry the dose history even under large edits.

```python
DR = R[R.event.isin(['insulin','norepinephrine']) & R.rate.notna()]
start_at = {}
for s_, hr_, ev_, rt_, ds_ in zip(DR.stay_id, DR.hour, DR.event, DR.rate, DR.dose):
    start_at.setdefault((int(s_), int(round(hr_*60))), []).append((ev_, float(rt_), float(ds_)))
def u_for(k, fI=1.0, fN=1.0):
    st = start_at.get((int(sid3[k]), int(round(float(hr3[k])*60))), [])
    if not st: return None
    atoms = [{'code':ev_, 'kind':0, 'rate':rt_*(fI if ev_=='insulin' else fN), 'dose':ds_*(fI if ev_=='insulin' else fN)} for ev_,rt_,ds_ in st]
    return eu2(atoms)[1].reshape(-1)
errs_u = [float((u_for(int(k)).double()-Uu[int(k)].double()).abs().max()) for k in drugb[:40] if u_for(int(k)) is not None]
print('u reproduction max err', round(max(errs_u),6), 'n', len(errs_u))
def q_of(cx):
    mu,_ = zprior2(cx); view = cx + lin2(mu,'z_head')
    return Fnn.layer_norm(view + T64('type_emb.weight')[1], (192,), T64('query_norm.weight'), T64('query_norm.bias'), 1e-5)
ratio_ = rat_b.values
def tgt_dev(s, h, s0, fI, fN):
    rI = RI.get((s,h),0.0)*(fI if h>=s0 else 1.0); g = GF.get((s,h), GL.get((s,h), np.nan))
    dI_ = rI - (float(np.ravel(pibar(np.array([g])))[0]) if g==g else 0.0)
    rN = RN.get((s,h),0.0)*(fN if h>=s0 else 1.0); rNp = RN.get((s,h-1),0.0)*(fN if h-1>=s0 else 1.0)
    m = MA.get((s,h), MC.get((s,h), np.nan)); bi = (np.searchsorted(bandsN, m, side='left')-1) if m==m else 2
    rh = rNp*float(ratio_[min(max(bi,0),5)])
    if rNp>0 and (rN>0 or (s,h) in RN): dN_ = rN - rh
    else: dN_ = 0.0
    return dI_, dN_
if max(errs_u) < 1e-3:
    rs = np.random.RandomState(1); FAC = [0,0.5,1,1,1.5,2,3]; QQ=[]; TT=[]; GG=[]; MET=[]
    t0=time.time()
    for s in stays_:
        ks = np.where(sid3==s)[0]
        for rep in range(4):
            s0 = int(rs.choice([6,8,10,12,14,16])); fI = float(rs.choice(FAC)); fN = float(rs.choice(FAC))
            Uc = Uu.clone().to(fa_.dtype)
            for k in ks:
                if hr3[k] >= s0 and (int(s), int(round(float(hr3[k])*60))) in start_at: Uc[k] = u_for(int(k), fI, fN).to(fa_.dtype)
            Hs_, Os_, _, _ = run_stay(list(ks), Ee.to(fa_.dtype), Uc, V)
            for j in range(1, len(ks)):
                k = ks[j]; d_ = INFO[k]
                if not any(c in d_ for c in codes4+['glucose_lab']): continue
                cx = ctx_of_state(Os_[j-1].double()[None], Hs_[j-1].double()[None], float(dts[k])).double()
                QQ.append(q_of(cx)[0].numpy()); t = int(round(float(hr3[k])))
                TT.append([v for kk in range(1,5) for v in tgt_dev(int(s), t-kk, s0, fI, fN)])
                GG.append([('glucose_finger' in d_) or ('glucose_lab' in d_), ('map_arterial' in d_) or ('map_cuff' in d_)])
                MET.append([int(s), rep, s0, fI, fN, t])
    QQ=np.array(QQ); TT=np.array(TT); GG=np.array(GG); MET=np.array(MET)
    np.savez('/work/probe_ds.npz', QQ=QQ, TT=TT, GG=GG, MET=MET)
    print('dataset', QQ.shape, 'time', round(time.time()-t0))
```