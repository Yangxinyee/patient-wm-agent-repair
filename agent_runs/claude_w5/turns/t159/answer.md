Three findings change the picture:
1. The `u` path is not drowned out in the prelude. At drug bundles the attention output from `u` has norm ~530, comparable to the flowed state (~760). So dose information must be lost somewhere else: either `u` hardly changes with the dose amount, or a later stage suppresses it.
2. At reading bundles, heads 1 and 3 still attend to the empty-action `u` token (~0.95). Head 0 attends to the reading's `e`, and head 2 to the slots. So any edit of the value path that reads `u` must give zero output for the empty-action `u`.
3. The direct recurrence is weak (`abar` ≈ 0.36) and most flow dims barely decay over an hour. The state is rebuilt each step through the prelude and blocks, so a written signal persists only through the model's own dynamics. That favours a "belief shift" edit (make the drug bundle move the state the way a lower glucose reading would) over a separate decaying memory.

Next I'll trace a dose doubling at insulin bundles stage by stage: Δu, Δnorm1(u), Δattention output, ΔE and Δy. That locates where the dose signal vanishes. If Δu is already tiny relative to |u|, the action encoder hardly distinguishes amounts, and the edit has to add a dose-proportional component to `u`'s path. I'll also print the normalisation notes from VOCAB.

```python
P='core.prelude.0.'
st_ = {k:[] for k in ['u','n1u','ao','E','y','out','cos_u']}
for sid in list(bundles_B.keys())[:30]:
    R = run_tf(sid)
    Mv=[r['e'] for r in R]; Mt=[r['t'] for r in R]
    for k,r in enumerate(R):
        if not any(a['code']=='insulin' and a['kind']==0 for a in r['ats']): continue
        e2,u2 = enc2(ins_atoms_scaled(r['ats'],2.0))
        n1a = LNt(r['u'][None], P+'norm1')[0]; n1b = LNt(u2[None], P+'norm1')[0]
        Va = W[P+'attn.kv.weight'][192:] ; Wo = W[P+'attn.out.weight']
        st_['u'].append(float((u2-r['u']).norm())); st_['cos_u'].append(float(torch.dot(u2,r['u'])/(u2.norm()*r['u'].norm())))
        st_['n1u'].append(float((n1b-n1a).norm())); st_['ao'].append(float((Wo@(Va@(n1b-n1a))).norm()))
        lo=max(0,k-127); Mv_=torch.stack(Mv[lo:k]+[e2]); Mage=torch.tensor([(r['t']-tt)*3600.0 for tt in Mt[lo:k+1]],dtype=torch.float64)
        o2,y2,z2,E2,hf2 = cstep(r['out_p'], r['h_p'], r['dt'], e2, u2, Mv_, Mage)
        st_['E'].append(float((E2-r['E']).norm()/r['E'].norm())); st_['y'].append(float((y2-r['y']).norm()/r['y'].norm())); st_['out'].append(float((o2-r['out']).norm()/r['out'].norm()))
print("insulin x2 at one bundle, median over", len(st_['u']), "bundles:")
print(" |du| %.3f (|u|~%.1f, cos %.4f) | |dnorm1(u)| %.3f | |d attn_out| (if att=1) %.1f" % (np.median(st_['u']), float(r['u'].norm()), np.median(st_['cos_u']), np.median(st_['n1u']), np.median(st_['ao'])))
print(" rel |dE| %.4f | rel |dy| %.5f | rel |dout| %.5f" % (np.median(st_['E']), np.median(st_['y']), np.median(st_['out'])))
print("numeric_note:", str(VOCAB['numeric_note'])[:300])
print("value_bin_note:", str(VOCAB['value_bin_note'])[:300])
print("numeric keys sample:", list(VOCAB['numeric'].keys())[:8] if isinstance(VOCAB['numeric'],dict) else type(VOCAB['numeric']))
```