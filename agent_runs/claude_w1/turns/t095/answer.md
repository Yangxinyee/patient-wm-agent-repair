Type index 1 and a gap measured in seconds are clearly the right choices. The leftover error of about 2 has no structural explanation left. I accept the head replica as approximate; final numbers will always be checked with the tool.

Now the actual design question. Doses reach the state only through the prelude attention: slot tokens attend to the action token u_k and the latent token z_in(z_k). Any edit that makes the state carry a clean "dose minus policy-expected dose" signal must pass through that door.

A rough calculation on my estimated effects also matters here. In a rollout, the model propagates its own drawn readings forward with persistence of about 0.8 per hour. So the direct one-step term it needs is concentrated on the last one or two infusions: about −2.1 and −0.8 mg/dL per unit/hour, against a total of −13.7 over 12 hours. A short memory may therefore be enough.

To see which door is cleanest, I will measure on insulin-start bundles and on reading bundles:
- how much each prelude head's slot queries attend to the u token and to the z token;
- how variable the prelude's output norm is, since that sets the gain stability of any channel through E.

What would support the plan: a head that puts most of its weight on u (or z) whenever a drug is present, plus a fairly stable norm. A linear dose channel through that head's values would then be feasible. What would argue against it: attention spread thinly and variably. I would then route the signal through the latent posterior instead.

```python
def prelude_attn(prev, B, k):
    dt = (B[k]['t']-B[k-1]['t'])*3600
    h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
    ctx = prev[:192] + LNn(h.mean(0),'core.out_norm') + mlp2n(np.array([np.log(dt)/10]),'gap')
    z = zpost_of(ctx, B[k]['e'], B[k]['u'])
    X = np.vstack([h + Wo['core.slots'], B[k]['e'][None], B[k]['u'][None], linn(z,'z_in')[None]])
    Xn = LNn(X,'core.prelude.0.norm1')
    q = Xn @ Wo['core.prelude.0.attn.q.weight'].T; kk = Xn @ Wo['core.prelude.0.attn.kv.weight'][:192].T
    qh = q.reshape(7,4,48).transpose(1,0,2); kh = kk.reshape(7,4,48).transpose(1,0,2)
    s = qh @ kh.transpose(0,2,1)/np.sqrt(48); s -= s.max(-1,keepdims=True); p = np.exp(s); p /= p.sum(-1,keepdims=True)
    X2 = X + mha(Xn, Xn, 'core.prelude.0.attn'); X2 = X2 + ffn(LNn(X2,'core.prelude.0.norm3'),'core.prelude.0.ffn')
    return p[:, :4, :], X2[:4].std(-1)
PA = {'ins':[], 'ne':[], 'read':[], 'none':[]}; SIG = {'ins':[], 'ne':[], 'read':[], 'none':[]}
for s in setA_ids[:15]:
    B = bundles_of_stay(tuA[tuA.stay_id==s], RU0); rows = np.where(sid3==s)[0]
    for k in range(1,len(B)):
        lab = B[k]['labels']
        typ = 'ins' if 'insulin' in lab else ('ne' if 'norepinephrine' in lab else ('read' if any(l in MUNIT for l in lab) else 'none'))
        p, sg = prelude_attn(X3[rows[k-1]], B, k); PA[typ].append(p.mean(1)); SIG[typ].append(sg)
for t in PA:
    if not PA[t]: continue
    A_ = np.array(PA[t]); S_ = np.array(SIG[t])
    print(t, len(A_), 'mean attn per head over [s0..s3, e, u, z]:')
    for hh_ in range(4): print('   head', hh_, np.round(A_[:,hh_,:].mean(0),3), ' sd(u,z):', np.round(A_[:,hh_,5:].std(0),3))
    print('   slot-output std: mean %.3f  cv %.3f' % (S_.mean(), S_.std()/S_.mean()))
```