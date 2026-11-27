`datom` takes code ids (8 = insulin, 9 = norepinephrine), so I can now build the amplification.

The plan for this run:
1. Drop the useless writer edit, so the model is back to the original.
2. Confirm my encoder reimplementation matches `u_rec` and the empty-bundle constant `U_EMPTY`.
3. On a dense grid of insulin and norepinephrine rates, including single-drug and empty bundles, compute mlp2's hidden layer and its pre-norm output. Each drug's rate-induced deviation is measured from a reference rate, set to the mean recorded rate in the tuning data.
4. Fit one ridge map per drug from the hidden layer to that deviation. The empty bundle is pinned to zero, so non-drug bundles are untouched.
5. Add (k−1) times each map to the last linear layer of `encode_action.mlp2`. Downstream, the model then sees each drug's rate deviation k times larger, along its own representation.

I start at k_insulin = 5 and k_norepinephrine = 10. The current responses are −2.15 and +0.34, against targets of about −13.8 and +4.6. This gives a first calibration point while staying moderate.

If the maps fit well (held-out R² near 1), the saved model's rate decoders should read roughly k times the deviation.

```python
import numpy as np, pandas as pd, torch
for n in list(edited()): drop_edit(n)
print('edited after drop:', edited())
W1a=WA['encode_action.mlp1.0.weight']; b1a=WA['encode_action.mlp1.0.bias']; W1b=WA['encode_action.mlp1.2.weight']; b1b=WA['encode_action.mlp1.2.bias']
W2a=WA['encode_action.mlp2.0.weight']; b2a=WA['encode_action.mlp2.0.bias']; W2b=WA['encode_action.mlp2.2.weight']; b2b=WA['encode_action.mlp2.2.bias']
gN=WA['encode_action.norm.weight']; bN=WA['encode_action.norm.bias']; sw=WA['encode_action.size.weight'][:,0]; sbz=WA['encode_action.size.bias']
def LNv(s):
    mu=s.mean(-1,keepdims=True); sd=np.sqrt(((s-mu)**2).mean(-1,keepdims=True)+1e-5); return (s-mu)/sd*gN+bN
def mlp1(V): return gelu(V@W1a.T+b1a)@W1b.T+b1b
def hid_s(m):
    hid=gelu(m@W2a.T+b2a); return hid, hid@W2b.T+b2b
for ri,rn in [(2.0,0.1),(0.0,0.0),(6.0,0.3)]:
    V=np.stack([np.asarray(datom(8,ri),float).ravel(),np.asarray(datom(9,rn),float).ravel()])
    print('check',ri,rn,'%.2e'%np.abs(enc_int(V)[3]-np.asarray(u_rec(ri,rn),float).ravel()).max())
hidE,sE=hid_s(sbz[None,:]); print('empty vs U_EMPTY %.2e'%np.abs(LNv(sE)[0]-np.asarray(U_EMPTY,float).ravel()).max())
tun=pd.read_csv('/data/tuning.csv'); print(tun.columns.tolist())
ref={}
cc=[c for c in tun.columns if c.lower() in ('code','code_name','event','name','code_id')]; rc=[c for c in tun.columns if 'rate' in c.lower()]
for cid,nm in [(8,'insulin'),(9,'norepinephrine')]:
    if cc and rc:
        x=pd.to_numeric(tun[(tun[cc[0]]==nm)|(tun[cc[0]]==cid)][rc[0]],errors='coerce').dropna()
        if len(x): ref[cid]=float(x.mean()); print(nm,'n',len(x),'mean %.3f median %.3f frac0 %.3f p95 %.3f max %.2f'%(x.mean(),x.median(),(x==0).mean(),x.quantile(.95),x.max()))
rrefI=ref.get(8,1.5); rrefN=ref.get(9,0.46); print('refs',rrefI,rrefN)
rI=np.round(np.arange(0,15.001,0.25),4); rN=np.round(np.arange(0,3.0001,0.01),4)
hI=mlp1(np.stack([np.asarray(datom(8,r),float).ravel() for r in rI])); hN=mlp1(np.stack([np.asarray(datom(9,r),float).ravel() for r in rN]))
hIr=mlp1(np.asarray(datom(8,rrefI),float).ravel()[None]); hNr=mlp1(np.asarray(datom(9,rrefN),float).ravel()[None])
c2=sw*np.log1p(2)+sbz; c1=sw*np.log1p(1)+sbz
hid2,s2=hid_s(((hI[:,None,:]+hN[None,:,:])/np.sqrt(2)+c2).reshape(-1,192))
_,sIref=hid_s((hIr+hN)/np.sqrt(2)+c2); _,sNref=hid_s((hI+hNr)/np.sqrt(2)+c2)
dI2=(s2.reshape(len(rI),len(rN),192)-sIref[None]).reshape(-1,192); dN2=(s2.reshape(len(rI),len(rN),192)-sNref[:,None]).reshape(-1,192)
h1I,s1I=hid_s(hI+c1); _,s1Ir=hid_s(hIr+c1); h1N,s1N=hid_s(hN+c1); _,s1Nr=hid_s(hNr+c1)
X=np.vstack([hid2,h1I,h1N,hidE]); X=np.hstack([X,np.ones((len(X),1))])
YI=np.vstack([dI2,s1I-s1Ir,np.zeros_like(s1N),np.zeros((1,192))]); YN=np.vstack([dN2,np.zeros_like(s1I),s1N-s1Nr,np.zeros((1,192))])
wts=np.ones(len(X)); wts[-1]=1000.
rng=np.random.default_rng(0); te=np.zeros(len(X),bool); te[:len(hid2)]=rng.random(len(hid2))<0.15
def ridge(X,Y,w,a):
    Xw=X*w[:,None]; A=X.T@Xw+a*np.eye(X.shape[1]); A[-1,-1]-=a; return np.linalg.solve(A,Xw.T@Y)
for nm,Y in [('I',YI),('N',YN)]:
    B=ridge(X[~te],Y[~te],wts[~te],1e-3); P=X[te]@B
    print(nm,'heldout R2 %.5f'%(1-((P-Y[te])**2).sum()/((Y[te]-Y[te].mean(0))**2).sum()),'empty out %.2e'%np.abs(X[-1]@B).max(),'|dev| typical %.3f'%np.sqrt((Y[:len(hid2)]**2).sum(1)).mean())
BI=ridge(X,YI,wts,1e-3); BN=ridge(X,YN,wts,1e-3)
print('s spread (rms dev from mean, per vec) %.3f'%np.sqrt(((s2-s2.mean(0))**2).sum(1)).mean(), 'mean |s| %.3f'%np.sqrt((s2**2).sum(1)).mean())
pI=np.load('/work/p_ins.npy'); pNv=np.load('/work/p_ne.npy'); print('decoder shapes',pI.shape,pNv.shape)
def set_amp(kI,kN,save=True):
    Wn=W2b+(kI-1)*BI[:768].T+(kN-1)*BN[:768].T; bn=b2b+(kI-1)*BI[768]+(kN-1)*BN[768]
    if save:
        save_edit('encode_action.mlp2.2.weight',torch.tensor(Wn,dtype=torch.float32)); save_edit('encode_action.mlp2.2.bias',torch.tensor(bn,dtype=torch.float32))
    return Wn,bn
Wn,bn=set_amp(5.0,10.0)
for ri,rn in [(0,0),(1.5,0),(3,0),(6,0.5),(rrefI,rrefN),(rrefI,1.0),(rrefI,0.0)]:
    V=np.stack([np.asarray(datom(8,ri),float).ravel(),np.asarray(datom(9,rn),float).ravel()])
    m=mlp1(V).sum(0)/np.sqrt(2)+c2; hid=gelu(m@W2a.T+b2a); u0=LNv(hid@W2b.T+b2b); u1=LNv(hid@Wn.T+bn)
    print('r=(%.2f,%.2f) decI %.2f->%.2f decN %.3f->%.3f |du| %.3f'%(ri,rn,pI@u0,pI@u1,pNv@u0,pNv@u1,np.linalg.norm(u1-u0)))
u0E=LNv(hidE@W2b.T+b2b); u1E=LNv(hidE@Wn.T+bn); print('empty |du| %.2e'%np.abs(u1E-u0E).max()); print('edited:',edited())
```