The NLL damage is almost all in readings taken before a stay's first drug bundle (547 readings).

| Pass | Edit | ΔNLL before first drug | ΔNLL after |
|---|---|---|---|
| fwd12 | v7 at full gain | +0.245 | +0.009 |
| fwd13 | v7 at 0.3 gain | +0.175 | −0.002 |
| fwd14 | v9 | +0.173 | +0.062 |

After the first drug bundle, v7 costs almost nothing, so v7 itself is sound but leaks badly before any drug is given. The added insulin effect also builds up over hours in the rollouts. For v7 at gain 1 it goes from −0.8 to −4.4, the same shape as the true effect.

If I can remove the pre-drug leak, a v7 gain of about 0.6–0.7 should cost under 0.002 nats. That would give insulin ×2 around −8 (truth −13).

My guess is that before any drug, the attention head carrying the new signal (head 1) reads non-drug tokens: slot, event or latent tokens. Its readout B is not zero on those tokens.

This run tests that guess:
- print the helper signatures and the source of the build functions;
- measure the head-1 readout (new and original components) on slot tokens, latent tokens and the constant drug token, before and after the first drug.

If the readouts are large on pre-drug slot or latent tokens, the guess is confirmed. If they are small, the cause lies elsewhere, for example in the event tokens.

```python
import inspect, numpy as np, torch
def sig(n):
    f=globals().get(n)
    if f is None: return None
    try: return n+str(inspect.signature(f))
    except Exception: return n+'(?)'
print('SIG:', ' | '.join(s for s in (sig(n) for n in ['prelude','prelude_b','uv','enc','zb','drug_atom','end_atom','feats','LN1','build6','build7','build8','chan_coord','br','chs','fr','lnw','mlp','tgt_from']) if s))
for n in ['build7','enc','uv','zb']:
    try:
        src=inspect.getsource(globals()[n]).split('\n')
        print(f'SRC {n}:', ' ;; '.join(l.strip()[:110] for l in src[:14] if l.strip()))
    except Exception as ex: print('nosrc', n, type(ex).__name__)
for n in ['tI5','tN5','c','rbar','qs','gp','kv3']:
    v=globals().get(n); print(n, None if v is None else (v if isinstance(v,(int,float)) else getattr(v,'shape','?')), end='; ')
print()
def LNx(x):
    try: y=LN1(x)
    except Exception: y=LN1(torch.tensor(x,dtype=torch.float32))
    return np.asarray(y.detach().numpy() if isinstance(y,torch.Tensor) else y, dtype=np.float64)
try: print('LN1 check', float(np.abs(LNx(u_const[None,:].astype(np.float32))[0]-LN1c).max()))
except Exception as ex: print('LN1 err', repr(ex)[:200])
st=np.load('/work/fwd_15/states_original.npz'); Xs_=np.asarray(st['x'],dtype=np.float64); sd_=np.asarray(st['stay_id'])
dr_=np.asarray(st['has_insulin']).astype(bool)|np.asarray(st['has_norepinephrine']).astype(bool)
new_=np.r_[True, sd_[1:]!=sd_[:-1]]; pre_=np.zeros(len(sd_),bool); started=False
for i in range(len(sd_)):
    if new_[i]: started=False
    if dr_[i]: started=True
    pre_[i]=not started
hprev=np.zeros((len(sd_),4,192)); zprev=np.zeros((len(sd_),32))
hprev[1:]=Xs_[:-1,192:960].reshape(-1,4,192); zprev[1:]=Xs_[:-1,960:992]
hprev[new_]=0; zprev[new_]=0
kvo=load_original('core.prelude.0.attn.kv.weight').numpy().astype(np.float64)[240:288]
kvc=load('core.prelude.0.attn.kv.weight').numpy().astype(np.float64)[240:288]
Mi=np.linalg.inv(M); Ao=Mi@(A.T@kvo); Ac=Mi@(A.T@kvc)
def fam(T, nm, m):
    L=LNx(T[m].reshape(-1,192).astype(np.float32))
    fn=L@Ac.T; fo=L@Ao.T
    print(f'{nm}: n{len(L)} f_new rms {np.round(np.sqrt((fn**2).mean(0)),3)} p99|.| {np.round(np.percentile(np.abs(fn),99,axis=0),2)} | orig-A rms {np.round(np.sqrt((fo**2).mean(0)),3)}')
try:
    Tsl=hprev+slots[None]; Tz=(zprev@zinW.T+zinB)[:,None,:]
    fam(Tsl,'slots pre',pre_); fam(Tsl,'slots post-nondrug',(~pre_)&(~dr_)); fam(Tsl,'slots drug',dr_); fam(Tsl,'slots first-bundle',new_)
    fam(Tz,'z pre',pre_); fam(Tz,'z post-nondrug',(~pre_)&(~dr_))
    fam(u_const[None,None,:].repeat(1,0),'u_const',np.array([True]))
    Fu=LN1u@Ac.T; print('drug u fit err rms', np.round(np.sqrt(((Fu-F)**2).mean(0)),3), 'F rms', np.round(np.sqrt((F**2).mean(0)),3))
except Exception as ex: print('fam err', repr(ex)[:300])
with open('/work/notes.md','a') as f:
    f.write('\n- run290 KEY: factual cost is ~all from readings BEFORE first drug bundle (n547): v7 s=1 pre +.245/post +.009; v7 s=.3 pre +.175/post -.0018; v9 pre +.173/post +.062.\n  => v7 post-drug cost model c(s)~ -.0124 s + .0214 s^2 (zero s~.58, s=.7 -> +.002). Need to kill the PRE-DRUG LEAK. v7(1,1) ins x2 added/h -0.8..-4.4 (accumulates like truth).\n- run291: diagnosing head-1 readout on slot/z/u_const tokens pre vs post.\n')
```