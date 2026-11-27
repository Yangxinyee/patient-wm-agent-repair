A clear picture from the prelude attention:

- **Slots read the action token almost only.** Head 2 puts 100% of its weight on the action token u_k in every bundle type. Heads 0 and 1 put about 97–99% on it when insulin is present.
- **Nothing else feeds the slots.** The latent token gets essentially zero attention, and the slots ignore each other.

So the dose reaches the state through a nearly fixed, state-independent path: the value vectors of the action token, copied into all four slots. That explains why the dose is present but used weakly downstream. It also suggests a simpler readout. The total last dose should be far easier to decode from the head's query than the innovation was (R² 0.24). The policy-expected dose can then come from the head's own "last glucose / MAP value" register, which holds exactly the reading the policy used.

To train such decoders I want many states, not just the 100 tool-provided stays. My replica is exact per step, so I can compute states for training stays locally. That needs the initial-state convention for the first bundle of a stay.

This run tests:
- a zero initial state with a few first-gap conventions, by matching the latent of bundle 0;
- a full free-running replica over whole stays, comparing every bundle with the tool's saved states, to make sure errors do not accumulate.

A near-zero error over whole stays means I can generate unlimited exact states locally. If bundle 0 doesn't match, I will look for another initial-state convention, for example learned slot embeddings.

```python
def init_ctx(dt0):
    return np.zeros(192) + Wo['core.out_norm.bias'] + mlp2n(np.array([np.log(dt0)/10]),'gap')
s = setA_ids[0]; B = bundles_of_stay(tuA[tuA.stay_id==s], RU0); rows = np.where(sid3==s)[0]
zt0 = X3[rows[0],960:992]
for dt0 in [1e-6, 1.0, 60.0, 600.0, 3600.0]:
    zp = zpost_of(init_ctx(dt0), B[0]['e'], B[0]['u'])
    print('dt0', dt0, 'relerr z0 %.4f' % (np.abs(zp-zt0).mean()/np.abs(zt0).mean()))
def run_stay(B, dt0=1.0, keep=True, edit=None):
    W = Wo
    h = np.zeros((4,192)); out = np.zeros(192); Ms = []; states = []
    for k in range(len(B)):
        dt = dt0 if k==0 else (B[k]['t']-B[k-1]['t'])*3600
        hd = h*np.exp(-dt*np.exp(W['core.flow.a']))
        ctx = out + LNn(hd.mean(0),'core.out_norm') + mlp2n(np.array([np.log(dt)/10]),'gap')
        z = zpost_of(ctx, B[k]['e'], B[k]['u'])
        X = np.vstack([hd + W['core.slots'], B[k]['e'][None], B[k]['u'][None], linn(z,'z_in')[None]])
        Xn = LNn(X,'core.prelude.0.norm1'); X = X + mha(Xn, Xn, 'core.prelude.0.attn')
        X = X + ffn(LNn(X,'core.prelude.0.norm3'), 'core.prelude.0.ffn')
        E = LNn(X[:4], 'core.prelude_norm')
        Ms.append((B[k]['t'], B[k]['e'])); ents = Ms[-128:]
        Mv = np.stack([e for _,e in ents]); age = np.array([(B[k]['t']-t)*3600 for t,_ in ents])
        MA = Mv + mlp2n(np.log1p(age)[:,None], 'core.age')
        x = hd + E
        for L in ['core.blocks.0.0','core.blocks.0.1']:
            xn = LNn(x, L+'.norm1'); x = x + mha(xn, xn, L+'.attn')
            x = x + mha(LNn(x, L+'.norm2'), MA, L+'.cross')
            x = x + ffn(LNn(x, L+'.norm3'), L+'.ffn')
        sd = softplus(W['core.delta_in']); abar = np.exp(-sd*np.exp(W['core.a_in']))
        hm = abar*hd + sd*(E @ W['core.b_in.weight'].T) + (x - (hd + E))
        y = hm @ W['core.c_out.weight'].T
        yn = LNn(y,'core.coda.0.norm1'); y = y + mha(yn, yn, 'core.coda.0.attn')
        y = y + ffn(LNn(y,'core.coda.0.norm3'), 'core.coda.0.ffn')
        h = y; out = LNn(y.mean(0),'core.out_norm')
        states.append(np.concatenate([out, h.reshape(-1), z]))
    return np.array(states)
import time
for s in setA_ids[:3]:
    t0 = time.time(); B = bundles_of_stay(tuA[tuA.stay_id==s], RU0); rows = np.where(sid3==s)[0]
    St = run_stay(B, dt0=1.0)
    rel = np.abs(St - X3[rows]).mean(1)/np.abs(X3[rows]).mean(1)
    print(s, len(B), 'free-run relerr: first %.4f  max %.4f  last %.4f   (%.2fs)' % (rel[0], rel.max(), rel[-1], time.time()-t0))
```