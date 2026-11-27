The teacher-forced check confirms the edit is neutralised inside a single bundle step. The lag-1 insulin slope is −0.64 at G = 0, −0.72 at G = 1, and back to −0.60 at G = 30. Norepinephrine behaves the same way. NLL at G = 30 equals baseline exactly, so a large injection leaves the z token effectively unattended.

The previous scan fits this:
- more insulin (positive residual) gave no extra effect;
- less insulin (negative residual) got stronger with G.

So the injection direction lowers the z token's attention logit when the residual is positive and raises it when negative. The fix is to make the injection direction key-neutral: orthogonal to the gradient of the prelude attention logits with respect to the z token.

Next I'll capture the prelude inputs at drug bundles with a temporary wrapper (restored afterwards). Then I'll compute the attention logits myself. I'll keep output to short statistics. If my computation reproduces the known pattern (heads 0–2 put almost all weight on u at drug bundles), I can use it to build the key-neutral subspace in the next run.

```python
print('edited', edited())
pn=[n for n in NAMES if n.startswith('core.prelude')]
print('prelude tensors:', [(n.replace('core.prelude.',''), tuple(SHAPES[n])) for n in pn])
CAPT=[]; _bz=bstep_z
def bstep_z(W,out,h,dt,e,u,Mt,ages,zfn=None,info=None):
    if info is not None:
        hf=h*torch.exp(-dt*torch.exp(W['core.flow.a']))
        g=bmlp(W,torch.tensor([[math.log(dt)/10]],dtype=T),'gap'); ctx=out+bln(W,hf.mean(1),'core.out_norm')+g
        CAPT.append(dict(hf=hf[0].clone(),e=e[0].clone(),u=u[0].clone(),ctx=ctx[0].clone(),info={k:float(v) for k,v in info.items()}))
    return _bz(W,out,h,dt,e,u,Mt,ages,zfn,info)
try:
    for s in stays[:40]: tf_stay_z(W_cur,BUN[s],None)
finally:
    bstep_z=_bz
print('captured drug bundles', len(CAPT), 'ins', sum(c['info']['hi']>0 for c in CAPT), 'ne', sum(c['info']['hn']>0 for c in CAPT))
```