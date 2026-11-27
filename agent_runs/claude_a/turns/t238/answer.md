The linear readouts from the slots' norm3(X1) do not work. The insulin and non-insulin gate values overlap (insulin minimum −0.43, non-insulin maximum 0.31), and the rate decoder reaches only R² 0.58. The large slot content swamps the copied dose information after the norm.

Earlier notes found the rate is cleanly decodable from the u token itself (R² 0.995), with prelude head 3 attending u at insulin bundles. So the signal should be staged from head 3's own 48-dim output, before it is mixed into the slot residual. Next I'll check that output directly:
- head 3's attention weight on the u token, by bundle type;
- a drug vs non-drug gate fitted on that 48-dim output;
- insulin and norepinephrine rate decoders fitted at drug bundles.

What would confirm the design: a clean gate margin and R² > 0.95 for both rates on head 3's output. If the gate overlaps there too, I need a different gate source.

```python
o=pr['o']; att=pr['att']            # o: B,H,7,48 ; att: B,H,7,7
o3=o[:,3,:4].numpy(); a3u=att[:,3,:4,5].numpy()
drug=ins|ne
for nm,mk in [('ins',ins),('NEonly',ne&~ins),('nodrug',~drug)]:
    print(nm, mk.sum(), 'att3->u mean %.3f min %.3f max %.3f'%(a3u[mk].mean(), a3u[mk].min(), a3u[mk].max()))
Xo=np.concatenate([o3.reshape(-1,48), np.ones((B*4,1))],1)
for gname,pos in [('drug',drug),('ins',ins)]:
    bg=ridge_fit(Xo,np.where(rep4(pos),1.,-1.),lam=1e-3); gg=(Xo@bg).reshape(B,4)
    print('gate',gname,'pos min %.3f neg max %.3f'%(gg[pos].min(), gg[~pos].max()))
md=rep4(drug)
for nm,r_ in [('ins',ri),('ne',rn)]:
    b_=ridge_fit(Xo[md], rep4(r_)[md], lam=1e-4); s_=(Xo@b_).reshape(B,4)
    on=r_>0; off=drug&~on; res=s_[on]-r_[on][:,None]
    print(nm,'R2 %.4f resid %.3f | other-drug |s| p99 %.3f | nodrug |s| p99 %.2f'%(1-res.var()/r_[on].var(), res.std(), np.percentile(np.abs(s_[off]),99), np.percentile(np.abs(s_[~drug]),99)))
```