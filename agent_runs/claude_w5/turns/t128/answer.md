I now have every input-encoding detail:
- z-scores are (x − median)/iqr, clipped to ±10; fine bin = 1 + number of edges below x.
- The code and event-kind index maps are known.
- END atoms in the data carry rate and dose, but the model ignores those (no attributes).
- `h_init` and `o_init` are stored initial states.

Next I write a fresh float64 implementation of the full step: encoders, ctx, posterior z, prelude, two recurrent layers with age-keyed memory, h update, coda. I check it teacher-forced: start each step from the saved true previous state and compare with the saved next state for one stay. Relative errors around 1e-6 confirm the implementation; larger errors point to a specific component.

```python
import torch, math, torch.nn.functional as F
W64 = {n: load(n).double() for n in NAMES}
UNIT = {'glucose_finger':0,'glucose_lab':5,'map_arterial':2,'map_cuff':2,'insulin':3,'norepinephrine':4}
RUNIT = {'insulin':2,'norepinephrine':3}
NUMT = {(it['code'],it['attribute']): it for it in VOCAB['numeric']}
DRUGS = ('insulin','norepinephrine')
def lin_(x,n):
    y = x @ W64[n+'.weight'].T; b = W64.get(n+'.bias'); return y if b is None else y+b
def ln_(x,n): return F.layer_norm(x,(x.shape[-1],),W64[n+'.weight'],W64[n+'.bias'],1e-5)
def mlp_(x,n): return lin_(F.gelu(lin_(x,n+'.0')),n+'.2')
def att_(xq,xkv,n):
    q = lin_(xq,n+'.q'); kv = lin_(xkv,n+'.kv'); k,v = kv[:,:192],kv[:,192:]
    q=q.view(-1,4,48).transpose(0,1); k=k.view(-1,4,48).transpose(0,1); v=v.view(-1,4,48).transpose(0,1)
    a = torch.softmax(q@k.transpose(1,2)/math.sqrt(48),-1)
    return lin_((a@v).transpose(0,1).reshape(-1,192), n+'.out')
def zs_(c,attr,x):
    it=NUMT[(c,attr)]; return float(np.clip((x-it['median'])/it['iqr'],-10,10))
def fb_(c,attr,x): return 1+int(np.sum(np.array(NUMT[(c,attr)]['fine_bin_edges'])<x))
def avec(a):
    c=a['code']; kd=a['kind']
    v = W64['embed.code.weight'][code_id[c]]+W64['embed.kind.weight'][kd]+W64['embed.ekind.weight'][code_ek[c]]+W64['embed.kq.weight'][0]
    nv = torch.zeros(6,dtype=torch.float64)
    if kd==0:
        if c in UNIT: v = v+W64['embed.categorical.unit.weight'][UNIT[c]]
        if c in RUNIT: v = v+W64['embed.categorical.rate_unit.weight'][RUNIT[c]]
        for j,attr in enumerate(('value','dose','rate')):
            if attr in a and a[attr] is not None:
                v = v+W64['embed.bins.'+attr+'.weight'][fb_(c,attr,a[attr])]; nv[2*j]=zs_(c,attr,a[attr]); nv[2*j+1]=1
    return v + W64['embed.numeric.weight']@nv + W64['embed.numeric.bias']
def senc(vecs,n):
    s = mlp_(torch.stack(vecs),n+'.mlp1').sum(0)/math.sqrt(len(vecs)) if vecs else torch.zeros(192,dtype=torch.float64)
    s = s + lin_(torch.tensor([math.log1p(len(vecs))],dtype=torch.float64),n+'.size')
    return ln_(mlp_(s,n+'.mlp2'),n+'.norm')
def enc2(atoms):
    act=[avec(a) for a in atoms if a['code'] in DRUGS and a['kind']==0]
    wor=[avec(a) for a in atoms if not (a['code'] in DRUGS and a['kind']==0)]
    return senc(wor,'encode'), senc(act,'encode_action')
def cstep(out,h,dt_s,e,u,Mv,Mage):
    hf = h*torch.exp(-dt_s*torch.exp(W64['core.flow.a']))
    ctx = out + ln_(hf.mean(0),'core.out_norm') + mlp_(torch.tensor([math.log(max(dt_s,1e-9))/10],dtype=torch.float64),'gap')
    zz = mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    T = torch.cat([hf+W64['core.slots'], e[None], u[None], lin_(zz,'z_in')[None]],0)
    n1=ln_(T,'core.prelude.0.norm1'); T = T+att_(n1,n1,'core.prelude.0.attn'); T = T+mlp_(ln_(T,'core.prelude.0.norm3'),'core.prelude.0.ffn')
    E = ln_(T[:4],'core.prelude_norm'); xb = hf+E
    Mem = Mv + mlp_(torch.log1p(Mage)[:,None],'core.age')
    for l in (0,1):
        p=f'core.blocks.0.{l}'
        n1=ln_(xb,p+'.norm1'); xb=xb+att_(n1,n1,p+'.attn')
        xb=xb+att_(ln_(xb,p+'.norm2'),Mem,p+'.cross'); xb=xb+mlp_(ln_(xb,p+'.norm3'),p+'.ffn')
    sp=F.softplus(W64['core.delta_in']); abar=torch.exp(-sp*torch.exp(W64['core.a_in']))
    hn = abar*hf + sp*lin_(E,'core.b_in') + (xb-(hf+E))
    x = hn@W64['core.c_out.weight'].T
    n1=ln_(x,'core.coda.0.norm1'); x=x+att_(n1,n1,'core.coda.0.attn'); y=x+mlp_(ln_(x,'core.coda.0.norm3'),'core.coda.0.ffn')
    return ln_(y.mean(0),'core.out_norm'), y, zz, E, hf
print(S.files)
kst = [k for k in S.files if S[k].ndim==1 and len(S[k])==len(Xs_)]
print({k: S[k][:2] for k in kst})
```