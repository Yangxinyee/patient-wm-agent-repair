Run 70 shows the insulin dose write in the raw state is almost the same vector in all four slots (cosine 0.98 to 1.0), about 11.75 units per U/h. The norepinephrine write is 23 units per mcg/kg/min, the same in three slots and different in slot 2. The two drug directions are anti-aligned (cosine about −0.5).

Along the insulin direction, natural reading-bundle changes have sd 54 and the level itself sd 56. Amplifying all writes along it would therefore also amplify a lot of non-dose variation and hurt the factual fit badly. Write amplification is not usable unless it is selective.

Before choosing a lever I need to know why input amplification was so sublinear: z-columns ×5 gave only a 1.4-fold response. Pass 5 holds states from that edited model and the original on the same stays, so I will measure:
- the per-unit-z state change the edit caused at insulin and norepinephrine bundles;
- how it compares with 4× the original model's natural per-dose write, both along the dose direction and overall;
- how much of the change persists to the next reading bundle.

A large state change with a small readout change would mean the heads are insensitive to it. A small state change would mean saturation upstream, in the action encoder or prelude.

```python
Ye_=xe_[:,192:960]; Oe_=xe_[:,0:192]
dYe=Ye_-Y_; dOe=Oe_-O_
for nm_,mask,rc,rD_ in [('ins',grp_['ins_only'],0,rD_i),('ne',grp_['ne_only'],1,rD_n)]:
    m=mask&ok_
    if nm_=='ins': zz_=(R_[m,0]-1.5)/1.0
    else: zz_=(R_[m,1]-0.46)/0.59
    X=np.column_stack([np.ones(m.sum()),zz_])
    B,_,_,_=np.linalg.lstsq(X,dYe[m],rcond=None)
    s_e=B[1]
    r=rD_.reshape(4,192).mean(0); rh=r/np.linalg.norm(r)
    se_m=s_e.reshape(4,192).mean(0)
    unit = 1.0 if nm_=='ins' else 0.59
    print(nm_,'n=%d; edit-induced slope per unit z: |slot-mean| %.2f, along dose dir %.2f; natural dose write per unit z along dir %.2f -> 4x = %.2f'%(m.sum(),np.linalg.norm(se_m),se_m@rh,np.linalg.norm(r)*unit,4*np.linalg.norm(r)*unit))
    print('   cos(edit slope, natural dose dir) = %.3f; |dYe| mean slot-mean %.2f; intercept along dir %.2f'%(se_m@rh/np.linalg.norm(se_m), np.linalg.norm(dYe[m].reshape(-1,4,192).mean(1),axis=1).mean(), B[0].reshape(4,192).mean(0)@rh))
    nxt=np.array([np.where(prev_==i)[0][0] if (prev_==i).any() else -1 for i in np.where(m)[0]])
    okn=nxt>=0
    Bn,_,_,_=np.linalg.lstsq(X[okn],dYe[nxt[okn]],rcond=None)
    print('   at next bundle: slope along dir %.2f (persistence %.2f)'%(Bn[1].reshape(4,192).mean(0)@rh, (Bn[1].reshape(4,192).mean(0)@rh)/(se_m@rh+1e-9)))
mrd2=mr
print('reading bundles: |dYe| mean %.3f (edit leaves readings? should be small except memory)'%np.linalg.norm(dYe[mrd2].reshape(-1,4,192).mean(1),axis=1).mean())
```