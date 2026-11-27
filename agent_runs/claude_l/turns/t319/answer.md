Run 291 didn't work either. The "virtual reading" directions had small norms (1.7 and 3.7 per unit) and almost no effect on attention scores. Yet the offline readouts stayed at the no-effect level (insulin ×2 about −3.0 on these 40 stays) at every gain, and NLL didn't move.

The same flat, gain-independent result appeared in run 289. That suggests a simpler explanation than "no leverage": `mkW3` may use `VD_` not only as the injected token direction but also to decode the dose residual from u. If so, replacing `VD_` with an unrelated direction makes the residual channel collapse to roughly zero, and nothing is injected at all.

So this run tests that directly on the recorded drug bundles. I'll compare the z residual channel's values with the original `VD_` and with the message direction, and check their correlation with the residual recomputed from `DEC2`/`PH`. Then I'll build the edit the safe way: `mkW3` with the original `VD_`, so the residual is computed correctly, and only then overwrite the two injection columns of `z_in.weight` with the message directions. I'll evaluate two gain pairs offline.

If the channel collapses under the swapped `VD_` but readouts now move with gain, the message route works and only needs calibrating. If readouts are still flat with a correct residual, the message direction has no leverage. In that case I'll settle on the best safe version of the original channel.

```python
VDN_msg={k_:np.array(v_) for k_,v_ in VDV.items()}
def zch(W,cl):
    x=torch.stack([td9(q['ctx']) for q in cl]); e=torch.stack([td9(q['e']) for q in cl]); u=torch.stack([td9(q['u']) for q in cl])
    return bmlp(W,torch.cat([x,e,u],-1),'z_post')[:,30:32],x,e,u
VD_.update({k_:v_.copy() for k_,v_ in VD_old.items()}); Wo_=mkW3(1.0,1.0,W0z)
VD_.update(VDN_msg); Wm_=mkW3(1.0,1.0,W0z)
VD_.update({k_:v_.copy() for k_,v_ in VD_old.items()})
for d_,j_ in [('ins',0),('ne',1)]:
    zo_,x_,e_,u_=zch(Wo_,grp9[d_]); zm_,_,_,_=zch(Wm_,grp9[d_])
    dec_=(torch.cat([u_,torch.ones(len(u_),1,dtype=T)],1)@td9(DEC2[d_]))-(torch.cat([x_,torch.ones(len(x_),1,dtype=T)],1)@td9(PH[d_]))
    print(d_,'chan |old| %.4f |msgVD| %.4f'%(float(zo_[:,j_].abs().mean()),float(zm_[:,j_].abs().mean())),'corr(old,DEC-PH) %.3f'%float(np.corrcoef(zo_[:,j_].numpy(),dec_.numpy())[0,1]),
          'sd old %.3f sd DEC-PH %.3f'%(float(zo_[:,j_].std()),float(dec_.std())))
def build_post(gi,gn,dirs):
    Wx_=dict(mkW3(gi,gn,W0z)); Wz_=Wx_['z_in.weight'].clone()
    Wz_[:,30]=2*gi*torch.tensor(dirs['ins']).to(Wz_.dtype); Wz_[:,31]=2*gn*torch.tensor(dirs['ne']).to(Wz_.dtype); Wx_['z_in.weight']=Wz_; return Wx_
st40=list(BUN.keys())[:40]
for gi,gn in [(3.0,8.0),(8.0,20.0)]:
    Wp_=build_post(gi,gn,VDN_msg); rr_=evalW(Wp_,st40); print('MSG-post',gi,gn,{kk:vv[0] for kk,vv in rr_.items()},'tfnll %.5f'%tfnll(Wp_,st40)[0])
with open('/work/notes.md','a') as f:
    f.write("\n- Run 285-286: G_ global = F.gelu (restored; was clobbered). mkW3 old VD_ gains 3/6/10 saturate (~-3.6,+2.2,+.7,-.9). Run 289: key-null LS dirs huge norm (1000+) -> no effect. "
            "Run 291: 'glucose/MAP message' dirs (VDV, |d|1.7/3.7) via VD_ swap -> no effect; suspect mkW3 decodes r with VD_. Run 292 tests post-hoc z_in column replacement (build_post).\n")
```