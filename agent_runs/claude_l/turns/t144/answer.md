The z posterior is still off by about 0.38, so ctx is not right yet. A gap scale of 1 is far worse, so log(dt)/10 is probably the right form, though the time unit inside the log is unclear.

I'll test combinations of three things on the same 5 stays:
- the time unit inside the gap's log (seconds, minutes, hours, log1p of seconds);
- applying the time flow to h or not;
- the flow's time unit.

A combination with error near 1e-5 fixes ctx.

```python
def zcheck2(s, gapf, flow_unit=1.0, use_flow=True):
    i0=row_start[s]; B_=BUN[s]; errs=[]
    for k in range(1,len(B_)):
        prev=Xs[i0+k-1]; out_p=torch.tensor(prev[:192]); h_p=torch.tensor(prev[192:960].reshape(4,192))
        dt=(B_[k][0]-B_[k-1][0])*60.0
        e,u=enc_eu2(B_[k][1])
        hf=h_p*torch.exp(-(dt/flow_unit)*torch.exp(W0['core.flow.a'])) if use_flow else h_p
        g=mlp(W0, torch.tensor([gapf(dt)],dtype=T),'gap')
        ctx=out_p+lnorm(W0,hf.mean(0),'core.out_norm')+g
        z=mlp(W0, torch.cat([ctx,e,u]),'z_post')[:32].detach().numpy()
        errs.append(np.abs(z-Xs[i0+k,960:992]).max())
    return np.array(errs)
gaps={'ls/10':lambda d: math.log(d)/10,'lmin/10':lambda d: math.log(d/60)/10,'lh/10':lambda d: math.log(d/3600)/10,'l1p/10':lambda d: math.log1p(d)/10,
      'l1pmin/10':lambda d: math.log1p(d/60)/10,'l1ph':lambda d: math.log1p(d/3600)}
for gn,gf in gaps.items():
    for fu,uf in [(1.0,True),(60.0,True),(3600.0,True),(1.0,False)]:
        er=np.concatenate([zcheck2(s,gf,fu,uf) for s in stays[:3]])
        print(gn,'flow_unit',fu,'flow',uf,'median %.2e max %.2e'%(np.median(er),er.max()))
```