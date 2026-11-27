Two results:
- `head_pred` is broken. It depends on a global that has since been overwritten with None, and I can't see its source.
- `drug_atoms` works. It returns the drug bundle's atom vectors (three for this bundle) before set encoding.

Rather than lean on opaque helpers, I'll write a small, transparent rebuild of the pieces I need, straight from the architecture text:
- the context and view from a stored state row: time flow, out_norm of the slot mean, the gap MLP, and the z prior mean through z_head;
- q via query_norm;
- the action set encoder, which gives u_k.

First I'll check q against Q99 using the reading table's state index `k`. I'll check both whether k is the reading's own row or the previous one, and which z convention Q99 used.

A near-exact match would confirm the rebuild and give me a way to compute Δq for edited-model states. If nothing matches, the gap input or the view's z convention differs, and I'll test variants.

```python
def LNf(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=((x-m)**2).mean(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*w+b
def mlpf(x,pre): return gelu(x@L(pre+'.0.weight').T+L(pre+'.0.bias'))@L(pre+'.2.weight').T+L(pre+'.2.bias')
fa_a=L('core.flow.a'); onw,onb=L('core.out_norm.weight'),L('core.out_norm.bias')
qnw,qnb=L('query_norm.weight'),L('query_norm.bias'); temb1=L('type_emb.weight')[1]
zhW,zhb=L('z_head.weight'),L('z_head.bias')
def view_from(prev,dt,zmode='prior'):
    prev=np.atleast_2d(prev); dt=np.atleast_1d(dt).astype(float)[:,None]
    out_p=prev[:,:192]; h=prev[:,192:960].reshape(-1,4,192)
    hf=h*np.exp(-dt[:,:,None]*np.exp(fa_a))
    g=mlpf(np.log(dt)/10,'gap'); ctx=out_p+LNf(hf.mean(1),onw,onb)+g
    zp=mlpf(ctx,'z_prior'); z=zp[:,:32] if zmode=='prior' else np.zeros((len(ctx),32))
    view=ctx+z@zhW.T+zhb
    return LNf(view+temb1,qnw,qnb), ctx, zp
X2=S2['x'].astype(np.float64); kk_=M4.k.values.astype(int)
print('S2 stay at k:', S2['stay_id'][kk_[:3]], S2['hour'][kk_[:3]], ' readings', M4.stay_id.values[:3], M4.hour.values[:3])
for off in [0,1]:
    for zm in ['prior','zero']:
        qm,_,_=view_from(X2[kk_-off], M4.dt.values, zm)
        print('off',off,zm,'max|q-Q99| %.4f mean %.5f'%(np.abs(qm-Q99).max(), np.abs(qm-Q99).mean()))
```