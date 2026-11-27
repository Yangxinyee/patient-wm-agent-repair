The fit failed: the channel's error is about 9 U/h at drug bundles. At bundles without drugs, head 0's output mixes slot and e values. Forcing the decoder to zero on those rows clashes with reading the rate from u, so a decoder that includes head 0 cannot be quiet there.

Head 2 is the natural fallback. It attends to u with weight 1.0 at every drug bundle and 0.95 elsewhere, where u is the fixed empty-set vector, so its output there is nearly constant. Its decoding error was 0.31 U/h, worse than heads 0+2 combined but probably acceptable.

This run fits the decoder on head 2 alone, with the same two kinds of rows. Before fitting I'll check the norms of the head outputs at the two bundle types, to see why the combined fit broke.
- If head 2 gives an error of about 0.3 U/h at drug bundles and close to 0 elsewhere, I'll use it.
- If not, I'll fall back to reading the rate from more of u's representation, using an explicit constraint for empty u.

```python
o02n=CN['o02'][~isD]; o02d=CN['o02'][isD]
print('norm head0 part: none %.2f drug %.2f | head2 part: none %.2f drug %.2f'%(np.linalg.norm(o02n[...,:48],axis=-1).mean(),np.linalg.norm(o02d[...,:48],axis=-1).mean(),np.linalg.norm(o02n[...,48:],axis=-1).mean(),np.linalg.norm(o02d[...,48:],axis=-1).mean()))
print('XdI norm head0 %.2f head2 %.2f'%(np.linalg.norm(XdI[:,:48],axis=1).mean(),np.linalg.norm(XdI[:,48:],axis=1).mean()))
for nm_,sl_ in [('head2',slice(48,96)),('head0+2',slice(0,96))]:
    for wn in [1.0,0.1,0.01]:
        Xd_=XdI[:,sl_]; Xn2=o02n.reshape(-1,96)[:,sl_]
        Xa=np.vstack([Xd_,np.sqrt(wn)*Xn2]); ya=np.r_[ydI,np.zeros(len(Xn2))]
        w_=np.linalg.solve(Xa.T@Xa+1e-4*np.eye(Xa.shape[1]),Xa.T@ya)
        eD=Xd_@w_-ydI; eN=Xn2@w_
        print(nm_,'w_none',wn,'drug err sd %.3f (ins rows %.3f, ne-only rows mean %.3f), none |val| mean %.3f p99 %.3f'%(eD.std(),eD[ydI>0].std(),eD[ydI==0].mean(),np.abs(eN).mean(),np.quantile(np.abs(eN),0.99)))
```