I now have the embedding tables needed for exact encoders.

This run rebuilds three pieces at the drug bundles of fwd_3, using the formulas in my notebook:
- the action encoder (u_k);
- the END-atom encoder (e_k);
- the prelude attention and FFN (tokens = decayed h + slots, e_k, u_k, z_in(z posterior)).

To check the replica, the attention pattern should reproduce my notebook: heads 1, 2 and 3 put about 0.99 on the u token, and head 0 puts about 0.95 on the slots.

I'll also decode the v3 key/value edit. An SVD of its head-1 value rows gives the three "a" directions. Regressing their outputs on [insulin rate, centred NE rate, insulin presence] identifies which is which. I'll also check that the original model's head-1 values have about zero component along them, and that they read about zero at the no-action token. If those hold, I can keep v3's clean readouts and only re-point the output projection.

```python
from scipy.special import erf
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def lnw(x, pre): return ln(x, L(pre+'.weight'), L(pre+'.bias'))
def mlp(x, pre): return gelu(x @ L(pre+'.0.weight').T + L(pre+'.0.bias')) @ L(pre+'.2.weight').T + L(pre+'.2.bias')
NUM = {(d['code'], d['attribute']): d for d in VOCAB['numeric']}; print('numeric keys', list(NUM.keys()))
EM = {k: L(f'embed.{k}.weight') for k in ['code','kind','ekind','kq']}
cat_u, cat_ru = L('embed.categorical.unit.weight'), L('embed.categorical.rate_unit.weight')
b_rate, b_dose, numW, numB = L('embed.bins.rate.weight'), L('embed.bins.dose.weight'), L('embed.numeric.weight'), L('embed.numeric.bias')
def zb(code, attr, x):
    d = NUM[(code, attr)]; return np.clip((x-d['median'])/d['iqr'],-10,10), 1+int(np.sum(np.array(d['fine_bin_edges'])<x))
def drug_atom(code, rate, dose):
    ci,ui,ri = (8,3,2) if code=='insulin' else (9,4,3); zr,br = zb(code,'rate',rate); zd,bd = zb(code,'dose',dose)
    return EM['code'][ci]+EM['kind'][0]+EM['ekind'][2]+cat_u[ui]+cat_ru[ri]+EM['kq'][0]+b_rate[br]+b_dose[bd]+numW@np.array([0,0,zd,1,zr,1.])+numB
def end_atom(code): return EM['code'][8 if code=='insulin' else 9]+EM['kind'][2]+EM['ekind'][2]+EM['kq'][0]+numB
def enc(atoms, pre):
    n=len(atoms); s = L(pre+'.size.bias') + L(pre+'.size.weight')[:,0]*np.log1p(n)
    if n: s = s + mlp(np.array(atoms), pre+'.mlp1').sum(0)/np.sqrt(n)
    return lnw(mlp(s[None], pre+'.mlp2')[0], pre+'.norm')
u_const = enc([], 'encode_action')
P = 'core.prelude.0.'; Wq, Wkv, Wo = L(P+'attn.q.weight'), L(P+'attn.kv.weight'), L(P+'attn.out.weight')
zinW, zinB = L('z_in.weight'), L('z_in.bias')
def prelude(Xs4, e, u, z, Wkv=Wkv, Wo=Wo):
    T = np.vstack([Xs4, e[None], u[None], (zinW@z+zinB)[None]]); N1 = lnw(T, P+'norm1')
    Q = N1[:4]@Wq.T; K = N1@Wkv[:192].T; V = N1@Wkv[192:].T; outs=[]; att=[]
    for h in range(4):
        sl=slice(48*h,48*h+48); s = Q[:,sl]@K[:,sl].T/np.sqrt(48); a = np.exp(s-s.max(1,keepdims=True)); a/=a.sum(1,keepdims=True); att.append(a); outs.append(a@V[:,sl])
    x1 = Xs4 + np.hstack(outs)@Wo.T; x2 = x1 + mlp(lnw(x1, P+'norm3'), P+'ffn'); return x1, x2, lnw(x2,'core.prelude_norm'), np.array(att), N1
tsel['ek'] = tsel.end_hour.round(4)
g_start = {k:v for k,v in tsel[tsel.event.isin(['insulin','norepinephrine'])].groupby(['stay_id','hk'])}
g_end = {k:v for k,v in tsel[tsel.event.isin(['insulin','norepinephrine'])].groupby(['stay_id','ek'])}
drows = np.where((s3['has_insulin']|s3['has_norepinephrine']) & prev_ok)[0]
REP = {k:[] for k in ['row','u','e','x1','x2','E','att','rI','rN']}
for k in drows:
    key=(sid[k], round(hr[k],4)); st = g_start.get(key); en = g_end.get(key)
    atoms = [drug_atom(r.event, r.rate, r.dose) for r in st.itertuples()] if st is not None else []
    ends = [end_atom(r.event) for r in en.itertuples()] if en is not None else []
    u = enc(atoms,'encode_action') if atoms else u_const; e = enc(ends,'encode')
    Xs4 = H[k-1]*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa))[None,:] + slots
    x1,x2,Ep,att,_ = prelude(Xs4, e, u, X[k,960:992])
    for nm,v in zip(['row','u','e','x1','x2','E','att'],[k,u,e,x1,x2,Ep,att]): REP[nm].append(v)
    REP['rI'].append(st[st.event=='insulin'].rate.sum() if st is not None else 0.); REP['rN'].append(st[st.event=='norepinephrine'].rate.sum() if st is not None else 0.)
REP = {k:np.array(v) for k,v in REP.items()}
att = REP['att']  # (n,4 heads,4 slots,7 tokens)
print('mean attention per head over tokens [slots, e, u, z]:')
for h in range(4): a = att[:,h].mean((0,1)); print(h, np.round([a[:4].sum(), a[4], a[5], a[6]],4), 'min on u', np.round(att[:,h,:,5].min(),4))
LN1u = lnw(REP['u'], P+'norm1'); LN1c = lnw(u_const, P+'norm1')
kv3 = np.load('/work/v3_g1_1_0.npz')['kv']; Dkv = (kv3 - Wkv)[240:288]
Uu, Ss, Vt = np.linalg.svd(Dkv); print('kv diff sv', np.round(Ss[:5],3)); A = Uu[:,:3]
zA = LN1u @ (A.T@Dkv).T; rI, rN = REP['rI'], REP['rN']; F = np.c_[rI, (rN-0.52)*(rN>0), (rI>0).astype(float)]
M = np.linalg.lstsq(F, zA, rcond=None)[0].T; res = zA - F@M.T
print('M (zA = M f)\n', np.round(M,4)); print('fit rms per comp', np.round(res.std(0),4), 'zA sd', np.round(zA.std(0),3))
print('zA at u_const', np.round(A.T@Dkv@LN1c,4), ' orig value along A at drug bundles: mean', np.round((LN1u@Wkv[240:288].T@A).mean(0),4), 'sd', np.round((LN1u@Wkv[240:288].T@A).std(0),4), 'at u_const', np.round(A.T@Wkv[240:288]@LN1c,4))
np.savez('/work/rep_drug.npz', **REP, A=A, M=M, u_const=u_const)
```