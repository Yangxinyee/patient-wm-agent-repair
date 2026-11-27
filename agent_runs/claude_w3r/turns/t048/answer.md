The bundles match exactly. My build_bundles gives the same 54 bundle times and atom counts as the model, END atoms included: 4 atoms at hour 1.083 are two starts plus two ENDs. So the grouping is right, and the z mismatch is in the arithmetic.

Next I will test the z posterior directly on this stay. For each bundle I take the previous out and h exactly from the states and compute ctx and z_post for a grid of variants:
- activation: GELU exact, GELU tanh, SiLU or ReLU;
- gap input scaling;
- unit-index mapping;
- whether e_k is zero when the bundle has no observations.

I score errors separately on action-only and observation-only bundles. If one variant drives the error to about 0, the input side is solved. If none does, the problem is elsewhere, for example z_post's output layout or END-atom attributes.

```python
import itertools, torch.nn.functional as Fnn
Wo = {n: torch.tensor(load_original(n)) for n in NAMES}
X0 = S_['x'][m0]; T0 = S_['hour'][m0]
def A_(x,k): return {'gelu':Fnn.gelu, 'gelut':lambda v:Fnn.gelu(v,approximate='tanh'), 'silu':Fnn.silu, 'relu':Fnn.relu}[k](x)
def L_(x,n,b=True): 
    y=x@Wo[n+'.weight'].T
    return y+Wo[n+'.bias'] if (b and n+'.bias' in Wo) else y
def MLP(x,n,ak): return L_(A_(L_(x,n+'.0'),ak),n+'.2')
def LN(x,n): return Fnn.layer_norm(x,(x.shape[-1],),Wo[n+'.weight'],Wo[n+'.bias'],1e-5)
def emb(a,o):
    v=Wo['embed.code.weight'][a['code']]+Wo['embed.kind.weight'][a['kind']]+Wo['embed.ekind.weight'][a['ek']]+Wo['embed.kq.weight'][0]
    v=v+Wo['embed.categorical.unit.weight'][a['unit']]+Wo['embed.categorical.rate_unit.weight'][a['ru']]
    num=torch.zeros(6)
    for j,(key,bn) in enumerate([('val','value'),('dose','dose'),('rate','rate')]):
        if a[key] is not None:
            c,x=a[key]; v=v+Wo['embed.bins.'+bn+'.weight'][fbin(c,bn,x)]; num[2*j]=zsc(c,bn,x); num[2*j+1]=1
    return v+Wo['embed.numeric.weight']@num+Wo['embed.numeric.bias']
def enc(X,n,ak):
    nn_=X.shape[0]; s=MLP(X,n+'.mlp1',ak).sum(0)/math.sqrt(nn_)+L_(torch.tensor([math.log(1+nn_)]),n+'.size')
    return LN(MLP(s,n+'.mlp2',ak),n+'.norm')
def zerr(o, ak, gk, ezero=True):
    bb=build_bundles(ev0,o); errs=[]; kinds=[]
    for k in range(1,len(bb)):
        out_=torch.tensor(X0[k-1,:192]); h_=torch.tensor(X0[k-1,192:960]).reshape(4,192)
        dt=(bb[k][0]-bb[k-1][0])*3600
        hf=h_*torch.exp(-dt*torch.exp(Wo['core.flow.a']))
        gin={'l10':math.log(dt)/10,'l':math.log(dt),'l1p10':math.log1p(dt)/10,'lmin10':math.log(dt/60)/10}[gk]
        ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([gin]),'gap',ak)
        at=bb[k][1]; ob=[a for a in at if a['ek']!=2]; ac=[a for a in at if a['ek']==2]
        e_=enc(torch.stack([emb(a,o) for a in ob]),'encode',ak) if ob else torch.zeros(192)
        u_=enc(torch.stack([emb(a,o) for a in ac]),'encode_action',ak) if ac else torch.zeros(192)
        zp=MLP(torch.cat([ctx,e_,u_]),'z_post',ak)[:32]
        errs.append(float((zp-torch.tensor(X0[k,960:])).abs().mean())); kinds.append(('A' if ac else '')+('O' if ob else ''))
    errs=np.array(errs); kinds=np.array(kinds)
    return {kk: round(float(np.median(errs[kinds==kk])),3) for kk in ['A','O','AO'] if (kinds==kk).any()}
UA = dict(unit={'map_cuff':2,'map_arterial':2,'glucose_finger':3,'glucose_lab':3,'insulin':4,'norepinephrine':5}, ru={'insulin':2,'norepinephrine':3}, end_attrs=True)
for ak in ['gelu','gelut','silu','relu']:
    for gk in ['l10','l','l1p10','lmin10']:
        print(ak, gk, zerr(UA, ak, gk))
```