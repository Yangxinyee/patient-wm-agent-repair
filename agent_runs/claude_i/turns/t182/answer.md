With the context recomputed from the tool's states, my new encoder and z_post reproduce the tool's z posterior exactly (max error 0.004). The stored `ctx` was stale. I now have an exact local encoder for arbitrary dose edits.

Next I need the full state update, to propagate an edited dose forward under teacher forcing. The kernel has `run_stay`, which takes a stay's bundle indices plus e and u arrays and rolls the core forward. I'll test it on two stays against the tool's states, using my exact e and u.

A relative error near 2e-7, as recorded in my notes, means I can simulate locally how a change at a drug bundle propagates to later states and predictions. If the error is large, I'll need to repair the step function first. I'll also save progress to the notebook.

```python
with open('/work/notes.md','a') as f_:
    f_.write('''
## Session 5 (restart after reply 164)
- fwd_3 states are TRAIN stays (1010...). Model rollout kernel (from fwd_1 de, OLS on lags): ins k(L) ~ -0.2,-0.21,-0.14,-0.15,-0.12,-0.16,-0.17,-0.07.. (true -2.0,-2.36,-2.1,-1.6,-1.3,-0.9,-0.6,-0.5,-0.4); ne k ~ +0.1..0.17 flat (true 3.2,2.5,1.7,0.9,0.3).
- Decodability (ridge CV, state after drug bundle): ins rate R2 .85/.77/.75/.67 (lags0-3), but DEVIATION (rate-pibar) only .24/.23/.14/0 -> state barely carries exogenous dose info. Head-only readout NOT viable; need dynamics edit.
- /work/enc2.py: exact encoder (emb_atoms, set_enc2, eu2(atoms)->e,u ; zpost2(ctx,e,u)). ctx2 recomputed from X3 states (out_prev + LN(mean h_prev*exp(-dt*exp(fa))) + gap2(dt), first bundle zero state) -> z err ~0. (old global ctx is WRONG/stale.)
''')
ks1 = list(np.where(S3==S3[0])[0])
try:
    WT
    Et = torch.tensor(Eall, dtype=WT['core.flow.a'].dtype); Ut = torch.tensor(Uall, dtype=WT['core.flow.a'].dtype)
    r_ = run_stay(ks1, Et, Ut)
    print(type(r_), len(r_) if hasattr(r_,'__len__') else '')
    Hs_r = r_[0]; Os_r = r_[1]
    Hs_r = torch.stack(Hs_r) if isinstance(Hs_r, list) else Hs_r
    Os_r = torch.stack(Os_r) if isinstance(Os_r, list) else Os_r
    eh = (Hs_r.reshape(len(ks1),-1).double().numpy() - X3[ks1,192:960]); eo = (Os_r.double().numpy()-X3[ks1,:192])
    print('h relerr', (np.abs(eh).max(1)/np.abs(X3[ks1,192:960]).max(1)).round(4)[:30])
    print('out maxerr', np.abs(eo).max(1).round(4)[:30])
except Exception as ex:
    import traceback; traceback.print_exc()
```