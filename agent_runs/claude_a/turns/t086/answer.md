Norepinephrine also looks linear in rate. The lag-1 slope per mcg/kg/min is about 3 to 3.5 across previous-rate bins from 0.11 to 0.99 (the lowest bin is noisy, 2.6 ± 0.6). The log-rate slope instead rises 6-fold over the same range, which rules out a log-dose effect. So for both drugs I'll treat the effect as linear in rate and additive. The x½ readout should then be about −½ of the x2 readout, with sel1 targets of roughly −14.3 / +7.2 (insulin) and +4.1 / −2.0 (norepinephrine).

Next I'll build the remaining offline pieces: the ctx function, the z-posterior mean, and the initial state at a stay's first bundle. I'll validate a fully self-driven teacher-forced replica that uses only the records as input, not the tool's states, against all sel2 states. Errors staying at float level across all bundles would confirm I can produce the 12-hour prefix of any stay and start rollouts from it. Growing errors would mean a missing convention, most likely at the first bundle.

```python
def mlpP(x, pre, P=None):
    P = Po if P is None else P
    return F.linear(F.gelu(F.linear(x, P[pre+'.0.weight'], P[pre+'.0.bias'])), P[pre+'.2.weight'], P[pre+'.2.bias'])
def ctx_fn(out_prev, h_prev, dt, P=None):
    P = Po if P is None else P
    hf = h_prev * torch.exp(-dt[:,None,None]*torch.exp(P['core.flow.a']))
    return out_prev + F.layer_norm(hf.mean(1),(192,),P['core.out_norm.weight'],P['core.out_norm.bias'],1e-5) + mlpP(torch.log(dt)[:,None]/10,'gap',P)
def zpost_mean(ctx, e, u, P=None): return mlpP(torch.cat([ctx,e,u],-1), 'z_post', P)[:,:32]
# check z_post on non-first rows using true previous states
nf_ = np.where(SB['prev'].notna().values)[0]; pv_ = SB['prev'].values[nf_].astype(int)
cx_ = ctx_fn(OUT[pv_], HS[pv_], torch.tensor(SB['dts'].values[nf_].astype(np.float32)))
zm_ = zpost_mean(cx_, eALL[nf_], uALL[nf_]); print("zpost err non-first", float((zm_-ZP[nf_]).abs().max()))
# first bundles: test h0=0,out0=0 with various dt0
f_ = np.where(SB['prev'].isna().values)[0][:20]
for dt0 in [1.0, 60.0, 3600.0, 1e-3]:
    cx0 = ctx_fn(torch.zeros(len(f_),192), torch.zeros(len(f_),4,192), torch.full((len(f_),), dt0))
    print("dt0", dt0, "zpost first err", float((zpost_mean(cx0, eALL[f_], uALL[f_]) - ZP[f_]).abs().max()))
y0, o0 = core_step(torch.zeros(4,192), eALL[f_[0]], uALL[f_[0]], ZP[f_[0]], 0.0, eALL[f_[0]][None], torch.zeros(1), {'km':'kvin','age':'raw'})
print("first-bundle state err (h0=0)", float((y0-HS[f_[0]]).abs().max()), float((o0-OUT[f_[0]]).abs().max()))
```