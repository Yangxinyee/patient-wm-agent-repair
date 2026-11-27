Stage 1 works, with a trade-off between fit and leakage:
- **Lightest penalty:** the u-token readout gives held-out R² 0.97 (insulin) and 0.94 (norepinephrine). Leakage at non-drug bundles is 0.38 and 0.14 rate units.
- **Heavier penalty:** leakage falls to about 0.05 at R² 0.94 and 0.84.
- In every case the readout is exactly 0 on the empty-drug token.

Leakage matters little, because the repurposed units will be GELU-gated by the drug/no-drug score. Leaked signal at non-drug bundles only enters through a tiny direct term, so I take the best-fitting readout.

Next I choose where in the slot space this signal is written. Head 1's output can reach a 48-dimensional subspace. Within it I pick the two directions with the least natural variation at drug bundles after the norm3 LayerNorm, by a generalised eigenproblem using the average LayerNorm Jacobian. A gated linear readout can then pick up a tiny signal without disturbing the rest.

I also save the design to the notebook.

```python
A_=Xd[Fd].T@Xd[Fd]+1.0*Xn[Fn_].T@Xn[Fn_]+1e-5*torch.eye(192,dtype=torch.float64)
QIN=Pp_@torch.linalg.solve(A_,Xd[Fd].T@Yd[Fd])     # 192x2 readout of rates from n1(u)
g3=W64['core.prelude.0.norm3.weight']; WO1=W64['core.prelude.0.attn.out.weight'][:,48:96]
def T_post_attn(r,at):
    e,u=enc2(at); hf=r['h0']*torch.exp(-r['dt']*torch.exp(W64['core.flow.a']))
    ctx=r['out0']+ln_(hf.mean(0),'core.out_norm')+mlp_(torch.tensor([math.log(max(r['dt'],1e-9))/10],dtype=torch.float64),'gap')
    zz=mlp_(torch.cat([ctx,e,u]),'z_post')[:32]
    Tk=torch.cat([hf+W64['core.slots'],e[None],u[None],lin_(zz,'z_in')[None]]); n1=ln_(Tk,'core.prelude.0.norm1')
    return (Tk+att_(n1,n1,'core.prelude.0.attn'))[:4]
Jbar=torch.zeros(192,48,dtype=torch.float64); N3=[]; cnt=0
for d in [d for d in D if isdrug(d) and d['sid'] in FIT][::6]:
    r=RK[d['sid']][d['k']]; Tp=T_post_attn(r,bundles_B[d['sid']][d['k']][1])
    for i in range(4):
        t_=Tp[i]; mu_=t_.mean(); sg=((t_-mu_)**2).mean().add(1e-5).sqrt(); xh=(t_-mu_)/sg
        M_=WO1-WO1.mean(0,keepdim=True); M_=M_-torch.outer(xh,xh@M_)/192
        Jbar+=g3[:,None]*M_/sg; cnt+=1
Jbar/=cnt
N3d=torch.cat([torch.cat([d['n3'+s] for s,_,_ in VK]) for d in D if isdrug(d) and d['sid'] in FIT]).double()
Sn=torch.cov(N3d.T)+1e-4*torch.eye(192,dtype=torch.float64)
Am=Jbar.T@torch.linalg.solve(Sn,Jbar); Bm=WO1.T@WO1
ev,evec=torch.linalg.eigh(torch.linalg.solve(Bm,Am))
cI_,cN_=evec[:,-1].real,evec[:,-2].real
print('top SNR eig',np.round(ev[-4:].numpy(),1),'median',float(ev.median()),'|d| per |c|: %.2f %.2f'%(float((WO1@cI_).norm()/cI_.norm()),float((WO1@cN_).norm()/cN_.norm())))
with open('/work/notes.md','a') as f:
    f.write("- Runs 162-178: cstep_h (hooked core, exact). Bundles: readings h:00, drugs h:05 (separate). Gate tests: dose not decodable from blocks-FFN hidden/E (R2<0.6); rates decodable from u token (R2 0.98/0.97); c_hat (E[rate|state], ridge on [out0,mean h0], R2 val .78 ins/.53 NE) decodable from prelude n3 (R2 .995/.97). Drug vs non-drug linearly separable (margin .84).\n- DESIGN: stage1 = rank-1 value edit in prelude attn head1 (reads rate from n1(u), zero on u0) writes tiny signal d into slots; stage2 = 3 repurposed smallest prelude-FFN units, GELU-gated by discriminant (twin cancels offset), compute G*(rate_signal - c_hat) at drug bundles; write w_out so that dh_new = kappa*v (v = mean belief direction: dh_new per 1 mg/dL lower glucose / 1 mmHg higher MAP). Head response: inj of v moves next mean -0.30/unit (glc), +0.22/unit (MAP).\n")
```