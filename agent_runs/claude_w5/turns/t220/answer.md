Zeroing prelude-FFN units 502, 131 and 140 changes the slot residual by about 0.035 against a norm of about 1,136 (relative 3e-5), so they are free to repurpose. Reading atoms carry a 'value' field, and the head rows H index readings by (sid, k).

Next, for about 80 training drug bundles, I compute two things:
1. The belief directions v_I and v_N: the change in h_new at the drug bundle per +1 mg/dL on the preceding glucose_finger reading, and per +1 mmHg on map_arterial. I use central ±5 differences, propagated through both steps and the memory.
2. The Jacobian of h_new with respect to an extra vector added to P in all slots.

A least-squares fit then gives one fixed P-space direction per drug that best reproduces each bundle's belief direction. A small relative residual (say under 30%) means a single write direction can carry the injection. A poor fit would push me to write at h_new instead.

```python
def mod_value(ats, code, dv):
    out, found = [], False
    for a in ats:
        if a.get('code') == code and 'value' in a:
            a2 = dict(a); a2['value'] = type(a['value'])(float(a['value']) + dv); out.append(a2); found = True
        else: out.append(a)
    return out, found
d_chk = steps_all[sids_B[3]][10]; print('Mv[-1]==e', float((d_chk['Mv'][-1]-d_chk['e']).abs().max()))
def hnew_mod(s, k, code, dv):
    st = steps_all[s]; dp, dk = st[k-1], st[k]
    ats_p, found = mod_value(bundles_B[s][k-1][1], code, dv)
    if not found: return None
    e_n, _ = enc2(ats_p)
    d1 = dict(dp); d1['e'] = e_n; M1 = dp['Mv'].clone(); M1[-1] = e_n; d1['Mv'] = M1
    o1 = mystep(d1)
    d2 = dict(dk); d2['out0'] = o1[0]; d2['h0'] = o1[1]; M2 = dk['Mv'].clone(); M2[-2] = e_n; d2['Mv'] = M2
    return mystep(d2)[5]
t0_ = time.time()
cand = [r for r in DB if r['sid'] in sid_tr]
rng_ = np.random.default_rng(0); pick = rng_.choice(len(cand), size=80, replace=False)
JS, VI, VN, SPb = [], [], [], []
for ii in pick:
    r = cand[ii]; s, k = r['sid'], r['k']
    hp = hnew_mod(s, k, 'glucose_finger', 5.0); hm_ = hnew_mod(s, k, 'glucose_finger', -5.0)
    mp_ = hnew_mod(s, k, 'map_arterial', 5.0); mm_ = hnew_mod(s, k, 'map_arterial', -5.0)
    d = steps_all[s][k]; R = {}; base_ = mystep(d, R=R)[5]
    J = torch.zeros(768, 192, dtype=torch.float64)
    for i in range(192):
        dP = torch.zeros(4, 192, dtype=torch.float64); dP[:, i] = 1.0
        J[:, i] = (mystep(d, dP=dP)[5] - base_).reshape(-1)
    JS.append(J.numpy()); SPb.append(float(R['P'].std(1).mean()))
    VI.append(None if hp is None else ((hp - hm_)/10).reshape(-1).numpy())
    VN.append(None if mp_ is None else ((mp_ - mm_)/10).reshape(-1).numpy())
print('time', round(time.time()-t0_,1), 'v_I avail', sum(v is not None for v in VI), 'v_N avail', sum(v is not None for v in VN))
def fit_w(VL):
    A = np.zeros((192,192)); bvec = np.zeros(192); idx = [i for i, v in enumerate(VL) if v is not None]
    for i in idx: A += JS[i].T@JS[i]; bvec += JS[i].T@VL[i]
    w = np.linalg.solve(A + 1e-6*np.trace(A)/192*np.eye(192), bvec)
    num = sum(((JS[i]@w - VL[i])**2).sum() for i in idx); den = sum((VL[i]**2).sum() for i in idx)
    cos = [float((JS[i]@w)@VL[i]/np.linalg.norm(JS[i]@w)/np.linalg.norm(VL[i])) for i in idx]
    gain = [float((JS[i]@w)@VL[i]/(VL[i]@VL[i])) for i in idx]
    return w, num/den, np.median(cos), np.percentile(cos, 10), np.mean(gain), np.std(gain)
wI_P, rI_, mcI, p10I, gI, sgI = fit_w(VI); wN_P, rN_, mcN, p10N, gN, sgN = fit_w(VN)
print('I: rel resid', round(rI_,3), 'cos med', round(mcI,3), 'p10', round(p10I,3), 'gain mean', round(gI,3), 'sd', round(sgI,3), '|w|', round(np.linalg.norm(wI_P),2))
print('N: rel resid', round(rN_,3), 'cos med', round(mcN,3), 'p10', round(p10N,3), 'gain mean', round(gN,3), 'sd', round(sgN,3), '|w|', round(np.linalg.norm(wN_P),2))
print('|v_I| mean', np.mean([np.linalg.norm(v) for v in VI if v is not None]).round(3), '|v_N| mean', np.mean([np.linalg.norm(v) for v in VN if v is not None]).round(3))
```