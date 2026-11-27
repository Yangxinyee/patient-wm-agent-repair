Scanning the gap input doesn't help: the best error is still 3.7. So the mismatch is not the first time gap. It is likely the activation function, the layout of the z_post output, or something in the encoder.

Next I'll test the activation (GELU, tanh-GELU, SiLU, ReLU) crossed with which half of z_post's output is the mean, and print the hour and atom count of the first state row as a sanity check. A near-zero error for one combination would identify it.

```python
print('first row hour', Z['hour'][r0[0]], 'n_atoms', Z['n_atoms'][r0[0]])
acts={'gelu':Fn.gelu,'gelu_tanh':lambda x:Fn.gelu(x,approximate='tanh'),'silu':Fn.silu,'relu':Fn.relu}
res_=[]
for an,af in acts.items():
    def mlpA(x,p,af=af): return lin(af(lin(x,p+'.0')),p+'.2')
    def encA(av,p):
        n_=av.shape[0]
        s_=mlpA(av,p+'.mlp1').sum(0)/np.sqrt(n_)+lin(torch.tensor([np.log(1+n_)],dtype=torch.float32),p+'.size')
        return lnp(mlpA(s_,p+'.mlp2'),p+'.norm')
    eA=encA(emb_visit(11,3,0,0)[None],'encode')
    for xg in np.linspace(-3,1,41):
        g_=mlpA(torch.tensor([[xg]],dtype=torch.float32),'gap')[0]
        o_=mlpA(torch.cat([g_,eA,torch.zeros(192)]),'z_post')
        for half in [0,1]:
            res_.append((round(float((o_[32*half:32*half+32]-zt).abs().max()),4),an,round(xg,2),half))
res_.sort(); print(res_[:6])
```