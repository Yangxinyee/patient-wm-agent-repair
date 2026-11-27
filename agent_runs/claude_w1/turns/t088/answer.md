The rebuilt encoder is exact: relative error 0.0000 on all bundle types for 25 stays. Two settings were needed: insulin units/hour → rate-unit row 2 and mcg/kg/min → row 3 (my notebook was right, the stale global was not), and an empty action set encoded through the size path rather than as zero. So I now have exact e_k, u_k, dt and latent means for every bundle.

Next is the one-step core replica, checked teacher-forced. From the tool's state after bundle k−1, I compute the state after bundle k and compare it with the saved one:
1. time-flow decay;
2. the prelude over [h + slot embeddings, e_k, u_k, z_in(z)];
3. the two recurrent layers (slot self-attention, cross-attention to the memory, FFN) with the input-gate update;
4. c_out followed by the coda layer.

The undocumented details are tested as a small grid:
- whether the age embedding is added before or after the key projection;
- the age unit (seconds, minutes or hours);
- whether the current bundle is already in the memory.

One variant with near-zero error on h and out would confirm the replica. I could then simulate rollouts locally and design edits before spending forward passes. If no variant works, the error pattern will show which stage is wrong, for example a correct prelude but wrong blocks.

```python
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)
def mha(xq, xkv, pre, k_add=None, kin_add=None):
    q = xq @ Wo[pre+'.q.weight'].T
    Wkv = Wo[pre+'.kv.weight']
    k = (xkv if kin_add is None else xkv + kin_add) @ Wkv[:192].T
    v = xkv @ Wkv[192:].T
    if k_add is not None: k = k + k_add
    nq, nk = q.shape[0], k.shape[0]
    qh = q.reshape(nq,4,48).transpose(1,0,2); kh = k.reshape(nk,4,48).transpose(1,0,2); vh = v.reshape(nk,4,48).transpose(1,0,2)
    s = qh @ kh.transpose(0,2,1)/np.sqrt(48); s = s - s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
    o = (p @ vh).transpose(1,0,2).reshape(nq,192)
    return o @ Wo[pre+'.out.weight'].T
def ffn(x, pre): return linn(gelu(linn(x, pre+'.0')), pre+'.2')
def core_step(prev, B, k, Mlist, mode):
    dt = (B[k]['t']-B[k-1]['t'])*3600
    h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
    ctx = prev[:192] + LNn(h.mean(0),'core.out_norm') + mlp2n(np.array([np.log(dt)/10]),'gap')
    z = zpost_of(ctx, B[k]['e'], B[k]['u'])
    X = np.vstack([h + Wo['core.slots'], B[k]['e'][None], B[k]['u'][None], linn(z,'z_in')[None]])
    X = X + mha(LNn(X,'core.prelude.0.norm1'), LNn(X,'core.prelude.0.norm1'), 'core.prelude.0.attn')
    X = X + ffn(LNn(X,'core.prelude.0.norm3'), 'core.prelude.0.ffn')
    E = LNn(X[:4], 'core.prelude_norm')
    # memory
    ents = [(B[j]['t'], B[j]['e']) for j in Mlist]
    if mode['cur']: ents = ents + [(B[k]['t'], B[k]['e'])]
    ents = ents[-128:]
    x = h + E
    if len(ents):
        Mv = np.stack([e for _,e in ents]); age = np.array([(B[k]['t']-t)*3600/mode['unit'] for t,_ in ents])
        A = mlp2n(np.log1p(age)[:,None], 'core.age')
    for L in ['core.blocks.0.0','core.blocks.0.1']:
        xn = LNn(x, L+'.norm1'); x = x + mha(xn, xn, L+'.attn')
        if len(ents):
            xn = LNn(x, L+'.norm2')
            if mode['age']=='kin': x = x + mha(xn, Mv, L+'.cross', kin_add=A)
            elif mode['age']=='kout': x = x + mha(xn, Mv, L+'.cross', k_add=A)
            else: x = x + mha(xn, Mv + A, L+'.cross')
        x = x + ffn(LNn(x, L+'.norm3'), L+'.ffn')
    sd = softplus(Wo['core.delta_in']); abar = np.exp(-sd*np.exp(Wo['core.a_in']))
    hm = abar*h + sd*(E @ Wo['core.b_in.weight'].T) + (x - (h + E))
    y = hm @ Wo['core.c_out.weight'].T
    yn = LNn(y,'core.coda.0.norm1'); y = y + mha(yn, yn, 'core.coda.0.attn')
    y = y + ffn(LNn(y,'core.coda.0.norm3'), 'core.coda.0.ffn')
    return y, LNn(y.mean(0),'core.out_norm')
RU0 = {'units/hour':2,'mcg/kg/min':3}
BB = {s: bundles_of_stay(tuA[tuA.stay_id==s], RU0) for s in setA_ids[:6]}
results_core = []
for age in ['kin','kout','both']:
  for unit in [1.0, 60.0, 3600.0]:
    for cur in [True, False]:
        mode = dict(age=age, unit=unit, cur=cur); eh=[]; eo=[]
        for s in setA_ids[:6]:
            B = BB[s]; rows = np.where(sid3==s)[0]
            for k in range(1, len(B), 3):
                y, o = core_step(X3[rows[k-1]], B, k, list(range(k)), mode)
                ht = X3[rows[k],192:960].reshape(4,192); ot = X3[rows[k],:192]
                eh.append(np.abs(y-ht).mean()/np.abs(ht).mean()); eo.append(np.abs(o-ot).mean()/np.abs(ot).mean())
        results_core.append((age, unit, cur, np.mean(eh), np.mean(eo)))
        print(age, unit, cur, 'relerr h %.4f out %.4f' % (np.mean(eh), np.mean(eo)))
```