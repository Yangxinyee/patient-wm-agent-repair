Pass 8 shows the z26 lever does nothing. Amplifying z26 deviations ×5 at drug bundles left all four readouts unchanged to within ±0.02. So although the posterior encodes dose in z26, the state update hardly uses it. The dose effect must reach the state through the action token u_k in the prelude, which I cannot observe. I will undo the four ops.

What I can observe is where that signal lands: the raw recurrent state y, four slots of 192, which persists between bundles and is read by the heads. Next I will measure, from the pass-5 states:
- the per-unit-dose change in y written at insulin-only bundles, by regressing Δy on the insulin rate with controls;
- whether some direction in y-space is written by dose much more strongly than by the natural updates at reading bundles. This is a generalized eigenproblem, held out by stay.

Such a direction would let me amplify the write along it, through the writers into h-space (`core.b_in.weight` 'out', for example), without disturbing reading updates. Finding none would rule out this route as well.

```python
clear_ops()
with open('/work/notes.md','a') as fh:
    fh.write("- PASS 8 (E1 z26 x5 at drug bundles via z_in): readouts unchanged (-1.83,+0.76,-0.50,-0.60). z->state pathway is NOT the dose carrier. Cleared.\n")
Y_=xo_[:,192:960]  # 4 slots raw h-space
O_=xo_[:,0:192]
order_=np.lexsort((hr_,sid_))
prev_=np.full(len(sid_),-1)
for a,b in zip(order_[:-1],order_[1:]):
    if sid_[a]==sid_[b]: prev_[b]=a
ok_=prev_>=0
dY=np.zeros_like(Y_); dY[ok_]=Y_[ok_]-Y_[prev_[ok_]]
dO=np.zeros_like(O_); dO[ok_]=O_[ok_]-O_[prev_[ok_]]
mi_=grp_['ins_only']&ok_&np.isfinite(R_[:,4])
Xd=np.column_stack([np.ones(mi_.sum()),R_[mi_,0],R_[mi_,2],R_[mi_,2]>0,R_[mi_,4]/50,na_[mi_]])
Bd,_,_,_=np.linalg.lstsq(Xd,dY[mi_],rcond=None)
rD=Bd[1]   # per U/h
resid=dY[mi_]-Xd@Bd
print('ins bundles n=%d; |rD|=%.3f; D sd %.2f'%(mi_.sum(),np.linalg.norm(rD),R_[mi_,0].std()))
mrd=mr&ok_
Sig_read=np.cov(dY[mrd].T); Sig_res=np.cov(resid.T)
for nmx,Sg in [('reading dY',Sig_read),('ins resid',Sig_res)]:
    ev_s=np.linalg.eigvalsh(Sg); print(nmx,'top eig',np.round(ev_s[-5:],2),'n eig>1e-3*max:',(ev_s>1e-3*ev_s[-1]).sum())
rhat=rD/np.linalg.norm(rD)
print('along rD: dose-induced sd %.3f, reading dY sd %.3f, ins resid sd %.3f'%(np.linalg.norm(rD)*R_[mi_,0].std(), np.sqrt(rhat@Sig_read@rhat), np.sqrt(rhat@Sig_res@rhat)))
# generalized: w ∝ (Sig_read+ridge)^-1 rD ; held-out by stay parity
for rg in [1e-3,1e-2,1e-1,1.0]:
    res_=[]
    for par in [0,1]:
        trn=(sid_%2==par); tst=~trn
        m1=mi_&trn; X1=np.column_stack([np.ones(m1.sum()),R_[m1,0],R_[m1,2],R_[m1,2]>0,R_[m1,4]/50,na_[m1]])
        B1,_,_,_=np.linalg.lstsq(X1,dY[m1],rcond=None)
        S1=np.cov(dY[mrd&trn].T)
        w=np.linalg.solve(S1+rg*np.trace(S1)/768*np.eye(768),B1[1]); w/=np.linalg.norm(w)
        m2=mi_&tst; X2=np.column_stack([np.ones(m2.sum()),R_[m2,0],R_[m2,2],R_[m2,2]>0,R_[m2,4]/50,na_[m2]])
        B2,_,_,_=np.linalg.lstsq(X2,dY[m2]@w,rcond=None)
        res_.append((B2[1]*R_[m2,0].std(), (dY[mrd&tst]@w).std()))
    print('ridge',rg,'HO dose-induced sd / reading sd:',[('%.3f/%.3f'%a) for a in res_])
```