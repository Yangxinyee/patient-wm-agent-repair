The gating holds. Head 1 puts 1.000 of its attention on the u token at drug bundles and 0.919 on the empty-drug token elsewhere. That empty token's norm1 output is exactly the same at every non-drug bundle.

So a value readout q with q·n1(u0) = 0 is exactly silent at non-drug bundles, except for the 8% of attention that falls on other tokens.

Next I will fit q_I and q_N by weighted ridge:
- **Drug bundles, all variants:** fit the rates from the u token.
- **Non-drug bundles:** push toward 0 the attention-weighted mixture of other tokens each slot receives.

Zero on u0 is enforced exactly by projecting it out. On held-out stays I will report R² of the delivered signal against the rate at drug bundles, and its RMS at non-drug bundles.

R² near 0.98 with leakage of about 0.01 rate units would confirm stage 1.

```python
uh=(u0n1/u0n1.norm()).double(); Pp_=torch.eye(192,dtype=torch.float64)-torch.outer(uh,uh)
Xd,Yd,Fd,Xn,Fn_=[],[],[],[],[]
for n1,A,ri,rn,dr,f in TOK:
    n1=n1.double(); A1=A[1].double()
    if dr: Xd.append(n1[5]); Yd.append([ri,rn]); Fd.append(f)
    else:
        a=A1.clone(); a[:,5]=0; Xn.append(a@n1); Fn_+= [f]*4
Xd=torch.stack(Xd)@Pp_; Yd=torch.tensor(Yd,dtype=torch.float64); Fd=torch.tensor(Fd)
Xn=torch.cat(Xn)@Pp_; Fn_=torch.tensor(Fn_)
for wn in [1.,10.,100.]:
    for lam in [1e-5,1e-4]:
        A_=Xd[Fd].T@Xd[Fd]+wn*Xn[Fn_].T@Xn[Fn_]+lam*torch.eye(192,dtype=torch.float64)
        Q=Pp_@torch.linalg.solve(A_,Xd[Fd].T@Yd[Fd])     # 192x2, Q.T@u0n1 = 0
        pd_=torch.stack([t[0][5].double() for t in TOK if t[4]])@Q
        v=~Fd; r2=[1-float(((pd_[v,j]-Yd[v,j])**2).mean()/Yd[v,j].var()) for j in (0,1)]
        lk=(torch.cat([ (lambda a: (a.__setitem__((slice(None),5),0) or a)@t[0].double())(t[1][1].double().clone()) for t in TOK if (not t[4]) and (not t[5])])@Q).pow(2).mean(0).sqrt()
        print('wn %g lam %g val R2 I %.3f N %.3f | leak rms I %.4f N %.4f | Q.u0 %.1e'%(wn,lam,r2[0],r2[1],float(lk[0]),float(lk[1]),float((u0n1.double()@Q).abs().max())))
```